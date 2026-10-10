"""CPU checks for media fidelity, handoff, and adapter isolation; no model loading."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import av
import numpy as np
import torch

import app
from core_worker import Engine, safe_vae_output
import matte_tools as tools


class BackgroundRemovalTests(unittest.TestCase):
    def test_core_inference_tensor_can_be_normalized_without_mutating_it(self):
        with torch.inference_mode():
            original=torch.tensor([-1.,0.,1.])
        fn=safe_vae_output(torch,lambda image:image.add_(1).div_(2).clamp_(0,1))
        result=fn(original)
        torch.testing.assert_close(result,torch.tensor([0.,.5,1.]))
        torch.testing.assert_close(original,torch.tensor([-1.,0.,1.]))
        self.assertFalse(torch.is_inference(result))
    @classmethod
    def setUpClass(cls):
        # Preserve generated evidence; never remove customer/generated files.
        cls.root = Path(tempfile.mkdtemp(prefix='spokesman-matte-check-'))
        cls.video = cls.root/'portrait.mp4'
        subprocess.run([tools.ffmpeg(),'-hide_banner','-loglevel','error','-n',
                        '-f','lavfi','-i','testsrc2=size=96x128:rate=24',
                        '-f','lavfi','-i','sine=frequency=440:sample_rate=48000',
                        '-t','0.708333333','-c:v','libx264','-pix_fmt','yuv420p',
                        '-c:a','aac',str(cls.video)],check=True,capture_output=True)
        cls.long_video=cls.root/'long.mkv'
        subprocess.run([tools.ffmpeg(),'-hide_banner','-loglevel','error','-n',
                        '-stream_loop','-1','-i',str(cls.video),'-frames:v','150',
                        '-an','-c:v','ffv1',str(cls.long_video)],check=True,capture_output=True)

    def test_chunk_padding_keeps_every_original_frame(self):
        for count in [1,8,9,144,145]:
            frames=np.arange(count,dtype=np.uint8)[:,None,None,None]
            padded=tools.padded_frames(frames)
            self.assertLessEqual(len(padded),145)
            self.assertEqual((len(padded)-1)%8,0)
            np.testing.assert_array_equal(padded[:count],frames)
            np.testing.assert_array_equal(padded[-1],frames[-1])
        w,h,cw,ch=tools.canvas({'width':1080,'height':1920},384)
        self.assertLess(w,h)
        self.assertEqual(w%32,0)
        self.assertEqual(h%32,0)
        self.assertAlmostEqual(cw/ch,1080/1920,places=2)
        chunks=list(tools.video_chunks(self.long_video))
        self.assertEqual([len(chunk) for chunk in chunks],[145,5])

    def test_one_click_transfer_selects_the_matching_tab(self):
        value,tabs=app.send_to_background(str(self.video))
        self.assertEqual(value,str(self.video))
        self.assertEqual(tabs['selected'],'background-removal')
        with self.assertRaises(Exception):
            app.send_to_background(None)

    def test_real_exports_preserve_frames_audio_and_transparency(self):
        work=self.root/'export'
        work.mkdir()
        meta=tools.prepare_video(self.video,work/'source.mkv',384,0,0)
        exporter=tools.MatteExports(work,self.root,'cpu-export',meta,transparent=True)
        count=0
        for rgb in tools.video_chunks(work/'source.mkv'):
            matte=np.zeros(rgb.shape,dtype=np.float32)
            matte[:,:,:meta['crop_w']//2,:]=1
            exporter.write(rgb,matte)
            count+=len(rgb)
        result=exporter.finish(self.video,0)
        self.assertEqual(result['frames'],count)
        for key in ['preview','transparent','mask']:
            self.assertTrue(Path(result[key]).is_file())
            with av.open(result[key]) as video:
                self.assertEqual(len(list(video.decode(video=0))),count)
                if key!='mask':
                    self.assertTrue(video.streams.audio)
        # Explicit VP9 alpha decoder proves alpha was encoded, not just a black background.
        pixels=subprocess.check_output([tools.ffmpeg(),'-hide_banner','-loglevel','error',
                                        '-c:v','libvpx-vp9','-i',result['transparent'],
                                        '-frames:v','1','-f','rawvideo','-pix_fmt','rgba','pipe:1'])
        rgba=np.frombuffer(pixels,dtype=np.uint8).reshape(meta['crop_h'],meta['crop_w'],4)
        self.assertGreater(rgba[:,:meta['crop_w']//2,3].mean(),250)
        self.assertLess(rgba[:,meta['crop_w']//2:,3].mean(),5)

    def test_matte_reuses_base_and_restores_previous_adapter(self):
        # The media/exports are real; only the untested GPU sampling nodes are simulated.
        engine=Engine.__new__(Engine)
        engine.model_root=self.root/'models'
        (engine.model_root/'loras').mkdir(parents=True)
        (engine.model_root/'loras'/tools.ALPHA_FILE).write_bytes(b'placeholder')
        previous=object()
        engine.model=engine.base_model=previous
        engine.active_lora_key=('previous-adapter',)
        engine.alpha_model=engine.alpha_model_key=None
        engine.torch=torch
        engine.video_vae=engine.audio_vae=engine.clip=object()
        engine.nodes=Mock()
        engine.nodes.CLIPTextEncode.return_value.encode.return_value=[[]]
        def empty(w,h,n,b):
            engine.test_length=n
            engine.test_size=(w,h)
            return ({'samples':torch.zeros(1,128,(n-1)//8+1,h//32,w//32)},)
        engine.EmptyLTXVLatentVideo=Mock()
        engine.EmptyLTXVLatentVideo.execute.side_effect=empty
        engine.LTXVConditioning=Mock()
        engine.LTXVConditioning.execute.return_value=('positive','negative')
        engine.LTXVAddGuide=Mock()
        engine.LTXVAddGuide.execute.side_effect=lambda p,n,v,l,*a,**k:(p,n,l)
        engine.LTXVEmptyLatentAudio=Mock()
        engine.LTXVEmptyLatentAudio.execute.return_value=({},)
        engine.LTXVConcatAVLatent=Mock()
        engine.LTXVConcatAVLatent.execute.return_value=({},)
        engine.LTXVSeparateAVLatent=Mock()
        engine.LTXVSeparateAVLatent.execute.return_value=({}, {})
        engine.LTXVCropGuides=Mock()
        engine.LTXVCropGuides.execute.return_value=('positive','negative',{})
        engine.sample=Mock(return_value={})
        def decode(*args,**kwargs):
            w,h=engine.test_size
            return (torch.full((engine.test_length,h,w,3),.5),)
        engine.nodes.VAEDecodeTiled.return_value.decode.side_effect=decode
        alpha=object()
        def select(cfg,event):
            engine.model=alpha
        engine.select_lora=Mock(side_effect=select)
        request={'video_path':str(self.video),'short_edge':384,
                 'output_dir':str(self.root),'transparent':False,'seed':1}
        with patch.object(tools,'adapter_ready',return_value=True):
            one=engine.generate_matte(request,lambda _:None)
            two=engine.generate_matte({**request,'video_path':str(self.long_video)},lambda _:None)
        self.assertIs(engine.model,previous)
        self.assertEqual(engine.active_lora_key,('previous-adapter',))
        self.assertIs(engine.alpha_model,alpha)
        engine.select_lora.assert_called_once()
        self.assertNotEqual(one['preview'],two['preview'])
        self.assertEqual(one['frames'],17)
        self.assertEqual(two['frames'],150)
        self.assertEqual(engine.sample.call_count,3)
        self.assertEqual(one['width'],96)
        self.assertEqual(one['height'],128)
        engine.nodes.CLIPTextEncode.return_value.encode.assert_called_with(engine.clip,'')
        engine.sample.side_effect=RuntimeError('sampling error')
        with patch.object(tools,'adapter_ready',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'sampling error'):
                engine.generate_matte(request,lambda _:None)
        self.assertIs(engine.model,previous)
        self.assertEqual(engine.active_lora_key,('previous-adapter',))


if __name__=='__main__':
    unittest.main()
