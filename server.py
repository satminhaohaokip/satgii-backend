"""
SATGII Universal Backend - Real Server Version
Only for content you own or have rights to use.
Run: pip install -r requirements.txt && uvicorn server:app --host 0.0.0.0 --port 8000
Requires: ffmpeg + ffprobe installed on server.
"""
import os, re, json, uuid, shutil, subprocess, threading, time, wave, struct
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

BASE = Path(__file__).parent
DOWNLOADS = BASE / "downloads"
PROJECTS = BASE / "projects"
JOBS = BASE / "jobs"
for p in [DOWNLOADS, PROJECTS, JOBS]:
    p.mkdir(exist_ok=True)

app = FastAPI(title="SATGII Universal Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- helpers ----------
def detect_platform(url: str):
    u = url.lower()
    if "youtube.com" in u or "youtu.be" in u: return "youtube"
    if "instagram.com" in u: return "instagram"
    if "facebook.com" in u or "fb.watch" in u or "fb.gg" in u: return "facebook"
    if "drive.google.com" in u or "docs.google.com" in u: return "google_drive"
    if re.search(r"\.(mp4|webm|mov|m4v)(\?|#|$)", u): return "direct"
    if u.startswith("http"): return "direct_maybe"
    return "unknown"

def extract_yt_id(url: str):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None

def extract_drive_id(url: str):
    m = re.search(r"/file/d/([A-Za-z0-9_-]+)", url)
    if m: return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None

def save_job(job_id, data):
    (JOBS / f"{job_id}.json").write_text(json.dumps(data, indent=2))
def load_job(job_id):
    f = JOBS / f"{job_id}.json"
    if not f.exists(): return None
    return json.loads(f.read_text())

def ffprobe_info(path: Path):
    try:
        out = subprocess.run(["ffprobe","-v","quiet","-print_format","json","-show_format","-show_streams",str(path)], capture_output=True, text=True, timeout=60)
        j = json.loads(out.stdout or "{}")
        fmt = j.get("format", {})
        duration = float(fmt.get("duration", 0) or 0)
        size = int(fmt.get("size", 0) or path.stat().st_size)
        vs = next((s for s in j.get("streams",[]) if s.get("codec_type")=="video"), {})
        return {"duration": duration, "width": int(vs.get("width",0) or 0), "height": int(vs.get("height",0) or 0), "size": size, "format": fmt.get("format_name","")}
    except Exception as e:
        return {"duration": 0, "width":0, "height":0, "size": path.stat().st_size if path.exists() else 0, "error": str(e)}

def audio_rms_per_sec(video_path: Path, workdir: Path):
    wav = workdir / "audio.wav"
    try:
        subprocess.run(["ffmpeg","-y","-i",str(video_path),"-vn","-ac","1","-ar","16000",str(wav)], capture_output=True, timeout=300)
        if not wav.exists(): return []
        with wave.open(str(wav), "rb") as w:
            n = w.getnframes(); rate = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth()
            raw = w.readframes(n)
        if sw != 2: return []
        total = len(raw)//2
        vals = struct.unpack(f"<{total}h", raw[:total*2])
        per_sec = []
        for s in range(0, (total//rate)+1):
            seg = vals[s*rate:(s+1)*rate]
            if not seg: per_sec.append(0); continue
            rms = (sum(x*x for x in seg)/len(seg))**0.5 / 32768.0
            per_sec.append(min(1.0, rms*3))
        return per_sec
    except Exception:
        return []

def frame_motion(video_path: Path, workdir: Path, duration: float):
    frames_dir = workdir / "frames"
    frames_dir.mkdir(exist_ok=True)
    try:
        subprocess.run(["ffmpeg","-y","-i",str(video_path),"-vf","fps=1,scale=32:18",str(frames_dir/"f%04d.jpg")], capture_output=True, timeout=600)
    except Exception:
        pass
    files = sorted(frames_dir.glob("*.jpg"))
    # Use PIL if available, else fallback to file size diff
    try:
        from PIL import Image
        import numpy as np
        prev = None; samples=[]
        audio = audio_rms_per_sec(video_path, workdir)
        for i, f in enumerate(files):
            try:
                img = Image.open(f).convert("RGB")
                arr = np.array(img, dtype=np.float32)
            except Exception:
                continue
            motion=0.2; cut=0
            if prev is not None:
                diff = abs(arr-prev).mean()
                motion = min(1.0, diff/30.0)
                cut = min(1.0, diff/8.0)
            prev = arr
            samples.append({"t": float(i), "motion": float(motion), "audio": float(audio[i] if i < len(audio) else 0.4), "cut": float(cut)})
        return samples
    except Exception:
        # Fallback without PIL: return audio only
        audio = audio_rms_per_sec(video_path, workdir)
        return [{"t": float(i), "motion":0.3, "audio": float(audio[i] if i < len(audio) else 0.4), "cut":0} for i in range(int(duration))]

def score_for_type(typ, win):
    def avg(k): return sum(x[k] for x in win)/max(1,len(win))
    m=avg("motion"); a=avg("audio"); c=avg("cut")
    first_a=win[0]["audio"] if win else 0; last_a=win[-1]["audio"] if win else 0
    rising=max(0,last_a-first_a)
    change=abs(last_a-first_a)+abs((win[-1]["motion"] if win else 0)-(win[0]["motion"] if win else 0))
    if typ=="Curiosity Gap": return rising*0.5 + (1-abs(m-0.45))*0.3 + a*0.2, f"Audio rising {rising:.2f} Motion {m:.2f}"
    if typ=="Emotional Peak": return a*0.8+m*0.2, f"Loudest audio RMS {a:.2f}"
    if typ=="Visual Punch": return m*0.85+c*0.15, f"Highest motion {m:.2f}"
    if typ=="Pattern Interrupt": return c*0.9+m*0.1, f"Biggest cut {c:.2f}"
    if typ=="Story Twist": return min(1,change)*0.7+a*0.3, f"Biggest change {change:.2f}"
    if typ=="Payoff Tease": return a*0.5+m*0.5, f"High audio {a:.2f} + motion {m:.2f}"
    return (a+m)/2, f"Audio {a:.2f} Motion {m:.2f}"

def build_hooks(samples, duration, hook_len=20, hook_count=5, filter_type="all"):
    all_types=["Curiosity Gap","Emotional Peak","Visual Punch","Pattern Interrupt","Story Twist","Payoff Tease"]
    wanted=all_types if filter_type=="all" else [filter_type]
    cands=[]
    for si in range(len(samples)):
        start=samples[si]["t"]; end=start+hook_len
        if end>duration: break
        win=[s for s in samples if start <= s["t"] < end]
        if len(win)<3: continue
        for ty in wanted:
            sc, det=score_for_type(ty, win)
            cands.append({"type":ty,"start":start,"end":end,"score":sc,"reason":det+f" {start:.1f}-{end:.1f}s","avgMotion":sum(x["motion"] for x in win)/len(win),"avgAudio":sum(x["audio"] for x in win)/len(win),"avgCut":sum(x["cut"] for x in win)/len(win)})
    cands.sort(key=lambda x: x["score"], reverse=True)
    picked=[]; used={}
    for c in cands:
        if len(picked)>=hook_count: break
        if filter_type=="all":
            max_per=2 if hook_count>len(all_types) else 1
            if used.get(c["type"],0)>=max_per: continue
        if any(abs(p["start"]-c["start"])<hook_len*0.6 for p in picked): continue
        picked.append(c); used[c["type"]]=used.get(c["type"],0)+1
    if len(picked)<hook_count:
        for c in cands:
            if len(picked)>=hook_count: break
            if any(p["start"]==c["start"] and p["type"]==c["type"] for p in picked): continue
            picked.append(c)
    picked.sort(key=lambda x: x["start"])
    return [{"title":p["type"],"start":p["start"],"end":p["end"],"score":p["score"],"reason":p["reason"],"avgMotion":p["avgMotion"],"avgAudio":p["avgAudio"],"avgCut":p["avgCut"]} for p in picked]

def analyze_video(path: Path, job_id, hook_len=20, hook_count=5, filter_type="all"):
    workdir = DOWNLOADS / job_id
    workdir.mkdir(exist_ok=True)
    info = ffprobe_info(path)
    duration = info["duration"]
    if duration<=0:
        raise Exception("ffprobe could not read duration. File may be corrupt or not a video.")
    # Update job progress
    j=load_job(job_id); j["progress"]=20; j["info"]=info; save_job(job_id,j)
    samples = frame_motion(path, workdir, duration)
    j=load_job(job_id); j["progress"]=80; j["samples"]=samples; save_job(job_id,j)
    hooks = build_hooks(samples, duration, hook_len, hook_count, filter_type)
    scenes = [s for s in samples if s["cut"]>0.6]
    is_vertical = info["height"]>info["width"]
    shorts = {"is_vertical": is_vertical, "duration": duration, "is_shorts": duration<=60 and is_vertical, "scenes": len(scenes)}
    return {"info": info, "samples": samples, "hooks": hooks, "shorts": shorts}

# ---------- models ----------
class DetectReq(BaseModel):
    url: str
class ImportReq(BaseModel):
    url: str
    confirm_authorized: bool = False
    access_token: str | None = None
    api_key: str | None = None
    hook_length: int = 20
    hook_count: int = 5
    hook_type: str = "all"
class TrimReq(BaseModel):
    start: float
    end: float

# ---------- routes ----------
@app.get("/")
def health():
    return {"ok": True, "service": "SATGII Universal Backend", "needs": "ffmpeg + ffprobe, yt-dlp, only for content you own or have rights to"}

@app.get("/api/health")
def health_real():
    # REAL check, runs ffmpeg, ffprobe, yt-dlp right now, no fake version strings
    import subprocess, shutil
    def run(cmd):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            first = (out.stdout or out.stderr or "").strip().splitlines()
            return {"found": True, "version_line": first[0] if first else "ok", "returncode": out.returncode}
        except Exception as e:
            return {"found": False, "error": str(e)}
    ffmpeg = run(["ffmpeg","-version"])
    ffprobe = run(["ffprobe","-version"])
    ytdlp = run(["yt-dlp","--version"])
    # Also check python yt_dlp module
    try:
        import yt_dlp
        ytdlp_py = yt_dlp.version.__version__
    except Exception as e:
        ytdlp_py = f"missing: {e}"
    all_ok = ffmpeg.get("found") and ffprobe.get("found")
    return {"ok": all_ok, "ffmpeg": ffmpeg, "ffprobe": ffprobe, "yt_dlp_cli": ytdlp, "yt_dlp_python": ytdlp_py, "message": "If ffmpeg found is True, FFmpeg is really working. If False, deploy failed, do not trust trim."}

@app.post("/api/detect")
def detect(req: DetectReq):
    p = detect_platform(req.url)
    return {"platform": p, "url": req.url, "youtube_id": extract_yt_id(req.url) if p=="youtube" else None, "drive_id": extract_drive_id(req.url) if p=="google_drive" else None, "note": "Detection only. Does not mean access granted."}

def do_import_job(job_id, req: ImportReq):
    try:
        j=load_job(job_id); j["status"]="downloading"; j["progress"]=5; save_job(job_id,j)
        if not req.confirm_authorized:
            raise Exception("You must confirm you own this video or have rights to use it. Tick confirm_authorized.")
        platform = detect_platform(req.url)
        video_path = DOWNLOADS / f"{job_id}.mp4"
        meta = {"platform": platform, "url": req.url}
        # --- Google Drive ---
        if platform=="google_drive":
            fid = extract_drive_id(req.url)
            if not fid: raise Exception("Could not extract Drive File ID")
            dl = f"https://drive.google.com/uc?export=download&id={fid}"
            if req.api_key:
                dl = f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media&key={req.api_key}"
            r = requests.get(dl, stream=True, timeout=120, allow_redirects=True)
            if r.status_code!=200: raise Exception(f"Drive download failed status {r.status_code}. File may be private. Share as Anyone with link or use Drive API with your own key.")
            with open(video_path,"wb") as f:
                for chunk in r.iter_content(1024*1024):
                    if chunk: f.write(chunk)
            meta["drive_id"]=fid
        # --- 【entity-Instagram¦canonical_name=Instagram】 / 【entity-Facebook¦canonical_name=Facebook】 via Graph API if token ---
        elif platform in ("【entity-instagram¦canonical_name=Instagram】","【entity-facebook¦canonical_name=Facebook】") and req.access_token:
            # Try to extract media id from URL? Graph needs media ID, user must provide token + we try to resolve via oEmbed? If no media ID, fall back to yt-dlp
            # For Graph, we need media ID. If URL contains /p/ or /reel/, we cannot get ID without API. So try yt-dlp as fallback with token as cookie is not supported.
            # Attempt Graph: if user passed access_token as "TOKEN|MEDIA_ID" we support it
            token = req.access_token
            media_id = None
            if "|" in token:
                token, media_id = token.split("|",1)
            if media_id:
                gr = requests.get(f"https://graph.facebook.com/v19.0/{media_id}?fields=id,media_type,media_url,caption&access_token={token}", timeout=30)
                gj = gr.json()
                if "media_url" not in gj: raise Exception(f"Graph API failed: {gj.get('error',{}).get('message','no media_url')}")
                mr = requests.get(gj["media_url"], stream=True, timeout=120)
                with open(video_path,"wb") as f:
                    for chunk in mr.iter_content(1024*1024):
                        if chunk: f.write(chunk)
                meta["graph"]=gj
            else:
                raise Exception("For 【entity-Instagram¦canonical_name=Instagram】/【entity-Facebook¦canonical_name=Facebook】, provide access_token as TOKEN|MEDIA_ID, or use File Import. Direct URL scraping is blocked by platform.")
        # --- YouTube / Direct / others via yt-dlp or direct download ---
        else:
            # Direct MP4: try direct download first (fast, no yt-dlp)
            if platform in ("direct","direct_maybe"):
                r = requests.get(req.url, stream=True, timeout=120, headers={"User-Agent":"Mozilla/5.0"})
                ct = r.headers.get("content-type","")
                if r.status_code==200 and ("video" in ct or req.url.lower().endswith((".mp4",".webm",".mov"))):
                    with open(video_path,"wb") as f:
                        for chunk in r.iter_content(1024*1024):
                            if chunk: f.write(chunk)
                    meta["method"]="direct_download"
                else:
                    # Fall back to yt-dlp for direct_maybe that is actually a page
                    raise Exception(f"Direct download failed status {r.status_code} content-type {ct}. Link is not a direct video file.")
            else:
                # YouTube, Instagram without Graph, Facebook without Graph -> yt-dlp
                # yt-dlp only for public or authorized content. If private, it will fail unless cookies/token provided.
                try:
                    import yt_dlp
                except ImportError:
                    raise Exception("yt-dlp not installed on server. pip install yt-dlp")
                ydl_opts = {"format":"best[ext=mp4]/best","outtmpl":str(video_path),"quiet":True,"no_warnings":True,"noplaylist":True}
                if req.access_token and platform=="youtube":
                    # YouTube OAuth: user can pass cookies via access_token as path? Not supported here, we use no cookies, public only.
                    pass
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(req.url, download=True)
                    meta["yt_info"]={"title":info.get("title"),"duration":info.get("duration"),"uploader":info.get("uploader"),"id":info.get("id")}
                if not video_path.exists():
                    # yt-dlp may have added extension
                    for ext in ["mp4","webm","mkv"]:
                        cand = DOWNLOADS / f"{job_id}.{ext}"
                        if cand.exists(): video_path=cand; break
                if not video_path.exists(): raise Exception("yt-dlp did not produce a file. Video may be private, age-gated, or blocked.")
        # --- Verify + Analyze ---
        j=load_job(job_id); j["status"]="analyzing"; j["progress"]=15; j["meta"]=meta; save_job(job_id,j)
        result = analyze_video(video_path, job_id, req.hook_length, req.hook_count, req.hook_type)
        # Save project
        proj_id = job_id
        proj = {"id": proj_id, "url": req.url, "platform": platform, "meta": meta, "info": result["info"], "hooks": result["hooks"], "samples": result["samples"], "shorts": result["shorts"], "video_file": str(video_path), "created": time.time()}
        (PROJECTS / f"{proj_id}.json").write_text(json.dumps(proj, indent=2))
        j=load_job(job_id); j["status"]="done"; j["progress"]=100; j["result"]=result; j["project_id"]=proj_id; save_job(job_id,j)
    except Exception as e:
        j=load_job(job_id); j["status"]="failed"; j["error"]=str(e); j["progress"]=100; save_job(job_id,j)

@app.post("/api/import")
def import_url(req: ImportReq, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:12]
    save_job(job_id, {"id": job_id, "status":"queued","progress":0,"url":req.url,"created":time.time()})
    # Use thread so it works even without BackgroundTasks in some deploys
    th = threading.Thread(target=do_import_job, args=(job_id, req), daemon=True); th.start()
    return {"job_id": job_id, "status":"queued", "note":"Poll GET /api/job/{job_id} for progress. No fake success, only done means real file obtained."}

@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    j=load_job(job_id)
    if not j: raise HTTPException(404,"Job not found")
    return j

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), hook_length: int=20, hook_count: int=5, hook_type: str="all"):
    job_id = uuid.uuid4().hex[:12]
    save_job(job_id, {"id":job_id,"status":"downloading","progress":5,"url":f"file://{file.filename}"})
    video_path = DOWNLOADS / f"{job_id}.mp4"
    with open(video_path,"wb") as f:
        while True:
            chunk = await file.read(1024*1024)
            if not chunk: break
            f.write(chunk)
    try:
        result = analyze_video(video_path, job_id, hook_length, hook_count, hook_type)
        proj = {"id":job_id,"url":f"file://{file.filename}","platform":"file","meta":{"filename":file.filename},"info":result["info"],"hooks":result["hooks"],"samples":result["samples"],"shorts":result["shorts"],"video_file":str(video_path),"created":time.time()}
        (PROJECTS / f"{job_id}.json").write_text(json.dumps(proj, indent=2))
        save_job(job_id, {"id":job_id,"status":"done","progress":100,"result":result,"project_id":job_id})
        return {"job_id":job_id,"project_id":job_id,"result":result}
    except Exception as e:
        save_job(job_id, {"id":job_id,"status":"failed","error":str(e)})
        raise HTTPException(400, str(e))

@app.get("/api/projects")
def list_projects():
    out=[]
    for f in PROJECTS.glob("*.json"):
        try:
            j=json.loads(f.read_text()); out.append({"id":j["id"],"url":j["url"],"platform":j["platform"],"duration":j["info"]["duration"],"hooks":len(j["hooks"]),"created":j["created"],"title":j.get("meta",{}).get("yt_info",{}).get("title") or j["url"]})
        except Exception: pass
    return sorted(out, key=lambda x: x["created"], reverse=True)

@app.get("/api/projects/{pid}/video")
def project_video(pid: str):
    f=PROJECTS / f"{pid}.json"
    if not f.exists(): raise HTTPException(404,"Project not found")
    j=json.loads(f.read_text()); vp=Path(j["video_file"])
    if not vp.exists(): raise HTTPException(404,"Video file missing")
    return FileResponse(str(vp), media_type="video/mp4", filename=f"{pid}.mp4")

@app.post("/api/projects/{pid}/trim")
def trim_project(pid: str, req: TrimReq):
    f=PROJECTS / f"{pid}.json"
    if not f.exists(): raise HTTPException(404,"Project not found")
    j=json.loads(f.read_text()); vp=Path(j["video_file"])
    out = DOWNLOADS / f"{pid}-trim-{int(req.start)}-{int(req.end)}.mp4"
    dur = req.end-req.start
    if dur<=0: raise HTTPException(400,"Invalid trim")
    r=subprocess.run(["ffmpeg","-y","-ss",str(req.start),"-i",str(vp),"-t",str(dur),"-c","copy",str(out)], capture_output=True, text=True)
    if r.returncode!=0: raise HTTPException(500, f"FFmpeg trim failed: {r.stderr[-500:]}")
    return FileResponse(str(out), media_type="video/mp4", filename=out.name)

@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    f=PROJECTS / f"{pid}.json"
    if f.exists(): f.unlink()
    for ext in ["mp4","webm"]:
        vp=DOWNLOADS / f"{pid}.{ext}"
        if vp.exists(): vp.unlink()
    return {"ok": True}

  
