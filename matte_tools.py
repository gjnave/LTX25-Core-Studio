"""Optional Alpha Gen adapter and bounded-memory video/matte exports."""
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import threading

import av
import numpy as np
from PIL import Image

ALPHA_REPO = 'Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen'
ALPHA_REVISION = '6184df14b1b560bf9b6447d3ee126bd7e4a88513'
ALPHA_FILE = 'ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors'
ALPHA_SIZE = 1308787472
ALPHA_SHA256 = 'd9e143f979e0756f0c83d772e6a8890f4c6b933db1f7d93fc06e25c179f3fd36'
DOWNLOAD_LOCK = threading.Lock()
MAX_CHUNK_FRAMES = 145

def adapter_ready(model_root):
    path = Path(model_root) / 'loras' / ALPHA_FILE
    return path.is_file() and path.stat().st_size == ALPHA_SIZE

def adapter_status(model_root):
    return ('Background Removal adapter installed.' if adapter_ready(model_root)
            else 'Optional download: 1.3 GB. Your existing LTX models are reused.')

def check_adapter_access(token=None):
    """Check authorization to the actual gated weights with HEAD; download no bytes."""
    from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_url
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    credential = (token or '').strip() or get_token()
    if not credential:
        return {'allowed':False,'status':'no_login',
                'message':'Sign in on the Alpha Gen model page and accept its access terms. '
                          'Then paste a Hugging Face read token from the same account below '
                          'and click Check access. An existing Hugging Face login on this PC can also be used.'}
    try:
        metadata = get_hf_file_metadata(hf_hub_url(ALPHA_REPO,ALPHA_FILE,revision=ALPHA_REVISION),
                                        token=credential,timeout=20)
    except GatedRepoError:
        return {'allowed':False,'status':'access_denied',
                'message':'This account does not yet have download access. Open the Alpha Gen '
                          'model page, accept the terms/request access, then click Check access again. '
                          'If access is pending, wait for approval. Use a token from that same account '
                          'with permission to read this gated model.'}
    except HfHubHTTPError as error:
        status = getattr(getattr(error,'response',None),'status_code',None)
        if status == 401:
            return {'allowed':False,'status':'invalid_token',
                    'message':'Hugging Face did not accept this login token. Create a read token '
                              'from the account granted model access, paste it below, and check again.'}
        if status == 403:
            return {'allowed':False,'status':'access_denied',
                    'message':'Download permission is missing. Accept access on the model page '
                              'and allow this token to read the gated model, then check again.'}
        return {'allowed':False,'status':'unavailable',
                'message':'Hugging Face could not check access right now. Try again shortly; '
                          'this does not mean your access request was rejected.'}
    except Exception:
        return {'allowed':False,'status':'unavailable',
                'message':'Could not connect to Hugging Face to check access. Check your connection '
                          'and try again. Your access may already be approved.'}
    if metadata.size != ALPHA_SIZE or metadata.etag != ALPHA_SHA256:
        return {'allowed':False,'status':'unexpected_file',
                'message':'Hugging Face returned an unexpected adapter file. Check for an app update before downloading.'}
    return {'allowed':True,'status':'authorized',
            'message':'Access confirmed by Hugging Face. You can now download the Background Removal adapter.'}

def download_adapter(model_root, token=None):
    if not DOWNLOAD_LOCK.acquire(blocking=False):
        raise ValueError('The adapter download is already running.')
    try:
        if adapter_ready(model_root):
            return adapter_status(model_root)
        access = check_adapter_access(token)
        if not access['allowed']:
            raise ValueError(access['message'])
        from huggingface_hub import hf_hub_download, get_token
        folder = Path(model_root) / 'loras'
        folder.mkdir(parents=True, exist_ok=True)
        try:
            path = Path(hf_hub_download(ALPHA_REPO, ALPHA_FILE, revision=ALPHA_REVISION,
                                       local_dir=str(folder), token=(token or '').strip() or get_token()))
        except Exception:
            raise ValueError('Access was confirmed, but the adapter download did not finish. '
                             'Check your connection and try Check access again, then Download. '
                             'Partial downloads are kept for retry.') from None
        with path.open('rb') as stream:
            checksum = hashlib.sha256()
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                checksum.update(chunk)
            digest = checksum.hexdigest()
        if path.stat().st_size != ALPHA_SIZE or digest != ALPHA_SHA256:
            raise ValueError('The adapter did not pass verification. Download it again before generating.')
        return 'Background Removal adapter downloaded and verified. Ready to generate.'
    finally:
        DOWNLOAD_LOCK.release()

def ffmpeg():
    found = shutil.which('ffmpeg')
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def inspect_video(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Upload a video and wait for its preview first.')
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('The uploaded file has no video track.')
        stream = container.streams.video[0]
        fps = stream.average_rate or stream.guessed_rate or Fraction(24)
        duration = float(stream.duration * stream.time_base) if stream.duration else (container.duration or 0) / 1e6
        first = next(container.decode(stream), None)
        if first is None:
            raise ValueError('The uploaded video could not be decoded.')
        width, height = stream.width, stream.height
        if abs(int(first.rotation or 0)) % 180 == 90:
            width, height = height, width
        return dict(width=width, height=height, fps=fps, duration=duration)

def canvas(meta, short_edge):
    w, h = meta['width'], meta['height']
    # Preserve aspect ratio; pad to the model's 32-pixel grid rather than stretch.
    scale = min(1.0, 1920 / max(w, h), 1088 / min(w, h))
    if short_edge:
        scale = min(scale, float(short_edge) / min(w, h))
    crop_w, crop_h = max(2, round(w * scale / 2) * 2), max(2, round(h * scale / 2) * 2)
    width, height = max(64, math.ceil(crop_w / 32) * 32), max(64, math.ceil(crop_h / 32) * 32)
    return width, height, crop_w, crop_h

def prepare_video(source, destination, short_edge, start, duration):
    meta = inspect_video(source)
    start, duration = float(start), float(duration)
    if not math.isfinite(start) or start < 0 or (meta['duration'] > 0 and start >= meta['duration']):
        raise ValueError('Start time must be inside the video.')
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('Duration must be positive, or zero for the whole remaining video.')
    width, height, crop_w, crop_h = canvas(meta, short_edge)
    filters = f'fps={meta["fps"]},scale={crop_w}:{crop_h}:flags=lanczos,pad={width}:{height}:0:0:black,setsar=1'
    args = [ffmpeg(), '-hide_banner', '-loglevel', 'error', '-n', '-ss', str(start), '-i', str(source)]
    if duration:
        args += ['-t', str(duration)]
    args += ['-map', '0:v:0', '-vf', filters, '-an', '-c:v', 'ffv1', '-pix_fmt', 'rgb24', str(destination)]
    subprocess.run(args, check=True, capture_output=True)
    return {**meta, 'width': width, 'height': height, 'crop_w': crop_w, 'crop_h': crop_h}

def video_chunks(path):
    with av.open(str(path)) as container:
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
            if len(frames) == MAX_CHUNK_FRAMES:
                yield np.stack(frames)
                frames = []
        if frames:
            yield np.stack(frames)

def padded_frames(frames):
    count = len(frames)
    target = max(9, 1 + math.ceil((count - 1) / 8) * 8)
    if target > MAX_CHUNK_FRAMES:
        raise ValueError('A matte pass cannot exceed 145 frames.')
    return np.concatenate([frames, np.repeat(frames[-1:], target-count, axis=0)]) if target > count else frames

def background_image(path, width, height):
    if path:
        from PIL import ImageOps
        with Image.open(path) as source:
            return np.asarray(ImageOps.fit(ImageOps.exif_transpose(source).convert('RGB'), (width,height))) / 255.
    yy, xx = np.indices((height,width))
    grey = np.where(((xx//24 + yy//24) % 2)[...,None], .25, .4)
    return np.repeat(grey,3,axis=-1)

class MatteExports:
    def __init__(self, work, output_root, prefix, meta, background=None, transparent=True):
        self.work, self.root, self.prefix = Path(work), Path(output_root), prefix
        self.meta = meta
        self.width, self.height = meta['crop_w'], meta['crop_h']
        self.fps, self.count = meta['fps'], 0
        self.background = background_image(background, self.width, self.height)
        self.containers = []
        for name in ['matte', 'preview']:
            container = av.open(str(self.work / (name+'.mp4')), 'w')
            stream = container.add_stream('libx264', rate=self.fps)
            stream.width, stream.height, stream.pix_fmt = self.width,self.height,'yuv420p'
            stream.options = {'crf':'18','preset':'fast'}
            self.containers.append((container,stream))
        self.alpha_process = None
        self.alpha_log = None
        if transparent:
            self.alpha_log = (self.work / 'transparent-encoder.log').open('wb')
            self.alpha_process = subprocess.Popen([
                ffmpeg(), '-hide_banner','-loglevel','error','-n',
                '-f','rawvideo','-pix_fmt','rgba','-s',f'{self.width}x{self.height}',
                '-r',str(self.fps),'-i','pipe:0','-an',
                '-c:v','libvpx-vp9','-pix_fmt','yuva420p','-auto-alt-ref','0',
                '-crf','30','-b:v','0','-deadline','good','-cpu-used','4',
                str(self.work/'transparent.webm')], stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,stderr=self.alpha_log)

    def write(self, rgb, matte):
        for source, mask in zip(rgb,matte):
            source = source[:self.height,:self.width].astype(np.float32)/255.
            mask = mask[:self.height,:self.width]
            if mask.ndim == 3:
                mask = mask[...,:3].mean(axis=-1)
            mask = np.clip(mask,0,1)[...,None]
            alpha = np.repeat(mask,3,axis=-1)
            preview = source*mask+self.background*(1-mask)
            for pixels, (container,stream) in zip([alpha,preview],self.containers):
                frame = av.VideoFrame.from_ndarray(np.uint8(np.clip(pixels,0,1)*255),'rgb24')
                frame.pts = self.count
                frame.time_base = 1/self.fps
                for packet in stream.encode(frame):
                    container.mux(packet)
            if self.alpha_process:
                rgba = np.concatenate([np.uint8(source*255), np.uint8(mask*255)],axis=-1)
                self.alpha_process.stdin.write(rgba.tobytes())
            self.count += 1

    def close(self, abort=False):
        for container,stream in self.containers:
            try:
                if not abort:
                    for packet in stream.encode():
                        container.mux(packet)
            finally:
                container.close()
        self.containers = []
        if self.alpha_process:
            if abort:
                self.alpha_process.terminate()
            else:
                self.alpha_process.stdin.close()
            try:
                code = self.alpha_process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.alpha_process.kill()
                self.alpha_process.wait()
                raise RuntimeError('Transparent-video encoding did not finish. See '+str(self.work/'transparent-encoder.log'))
            finally:
                self.alpha_log.close()
            self.alpha_process = None
            if code and not abort:
                raise RuntimeError('Transparent-video encoding failed. See '+str(self.work/'transparent-encoder.log'))

    def finish(self, original, start):
        if not self.count:
            raise ValueError('No video frames were found in the selected range.')
        self.close()
        duration = self.count/float(self.fps)
        mask = self.root/(self.prefix+'-mask.mp4')
        shutil.copy2(self.work/'matte.mp4',mask)
        result = {'mask':str(mask), 'frames':self.count, 'width':self.width,'height':self.height,'fps':float(self.fps)}
        for name, ext, audio_codec in [('preview','mp4','aac'),('transparent','webm','libopus')]:
            source = self.work/(name+'.'+ext)
            if not source.is_file():
                result[name] = None
                continue
            output = self.root/(self.prefix+'-'+name+'.'+ext)
            args = [ffmpeg(),'-hide_banner','-loglevel','error','-n','-i',str(source),
                    '-ss',str(start),'-i',str(original),'-map','0:v:0','-map','1:a:0?',
                    '-t',str(duration),'-c:v','copy','-c:a',audio_codec,str(output)]
            subprocess.run(args,check=True,capture_output=True,timeout=180)
            result[name] = str(output)
        return result
