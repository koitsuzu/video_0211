import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from moviepy import VideoFileClip
from mistralai import Mistral

# 載入環境變數
load_dotenv()

def extract_audio(video_path: Path, audio_path: Path):
    """從影片提取音訊並儲存為 mp3"""
    print(f"正在從 {video_path.name} 提取音訊...")
    video = VideoFileClip(str(video_path))
    video.audio.write_audiofile(str(audio_path), logger=None)
    video.close()
    print(f"音訊已提取至 {audio_path}")

def transcribe_with_mistral(client: Mistral, audio_path: Path):
    """
    呼叫 Mistral API 進行轉錄，取得帶時間軸的 segment。
    """
    print(f"正在呼叫 Mistral API 進行轉錄...")
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.complete(
            model="voxtral-mini-latest",
            file={
                "content": f.read(),
                "file_name": audio_path.name
            },
            timestamp_granularities=["segment"]
        )
    return response

def load_terms(video_name: str = ""):
    """
    載入外部字詞庫，依影片名稱自動匹配對應的字詞組。
    匹配邏輯：檢查 terms.json 中的 key 是否出現在影片檔名中。
    若無匹配則使用 'default'。
    """
    terms_path = Path("terms.json")
    if not terms_path.exists():
        return {"corrections": {}, "key_terms": [], "topic_hint": ""}
    
    with open(terms_path, "r", encoding="utf-8") as f:
        all_terms = json.load(f)
    
    # 依影片名稱匹配字詞庫
    for key, terms in all_terms.items():
        if key == "default":
            continue
        if key in video_name:
            print(f"字詞庫匹配：「{key}」")
            return terms
    
    # 無匹配則用 default
    print("字詞庫匹配：使用預設 (default)")
    return all_terms.get("default", {"corrections": {}, "key_terms": [], "topic_hint": ""})

def process_and_summarize(client: Mistral, transcription_response, video_name: str = ""):
    """
    使用 Mistral Chat API 對逐字稿進行：
    1. 翻譯為繁體中文
    2. 依字詞庫校正專有名詞
    3. 篩選「關鍵知識點 (Key Knowledge Points)」
    4. 為每個知識點產生標題
    5. 產生內容摘要
    """
    print("正在處理文本：翻譯、校正專有名詞並篩選關鍵知識點...")
    
    terms = load_terms(video_name)
    
    segments = transcription_response.segments
    text_to_process = "\n".join([f"[{s.start}-{s.end}] {s.text}" for s in segments])
    
    # 動態組裝字詞庫提示
    terms_section = ""
    if terms.get("topic_hint"):
        terms_section += f"- 本影片主題：{terms['topic_hint']}\n"
    if terms.get("corrections"):
        correction_rules = "、".join([f"「{k}」→「{v}」" for k, v in terms["corrections"].items()])
        terms_section += f"- 名詞校正規則：{correction_rules}\n"
    if terms.get("key_terms"):
        terms_section += f"- 領域關鍵詞彙：{', '.join(terms['key_terms'])}\n"
    
    prompt = f"""
你是一個專業的影音逐字稿翻譯與教學重點摘要專家。
請將以下帶有時間軸的逐字稿內容進行精煉處理。

### 原始逐字稿內容：
{text_to_process}

### 字詞庫與專業術語參考：
{terms_section}

### 重要任務與要求：
1. **翻譯與校正**：將所有內容翻譯為「繁體中文」。請嚴格依照上方「名詞校正規則」修正錯誤用詞。
2. **篩選重點**：原始內容可能包含過多零碎的對話或雜訊。請從中挑選出「真正的關鍵知識點 (Key Knowledge Points)」。
3. **摘要**：提供一份整體的繁體中文內容摘要。
4. **輸出格式**：必須為 JSON。
5. **JSON 結構**：
{{
  "summary": "這裡填寫整體的繁體中文摘要",
  "key_moments": [
    {{
      "title": "此段落的精簡標題（5-15字）",
      "start": 0.0,
      "end": 10.5,
      "text": "翻譯並校正後的繁體中文內容"
    }},
    ...
  ]
}}
6. **準則**：
   - 請將鄰近且主題相同的 segment 合併為一個 key_moment，確保總數量適中（建議 5-15 個）。
   - 每個 key_moment 必須有一個精簡的「title」欄位，用一句話概括該段落的核心知識點。

請只返回 JSON 內容。
"""

    chat_response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    return json.loads(chat_response.choices[0].message.content)

def capture_screenshots(video_path: Path, segments: list, screenshot_dir: Path):
    """
    針對每個 key_moment 擷取最具代表性的畫面。
    策略：在每個時間段內取多個候選幀，選取「視覺內容最豐富」的一幀。
    """
    print(f"正在智慧擷取截圖至 {screenshot_dir}...")
    if screenshot_dir.exists():
        import shutil
        shutil.rmtree(screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    
    video = VideoFileClip(str(video_path))
    
    screenshot_paths = []
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        duration = end - start
        
        # 在段落內取多個候選時間點 (最多 5 個)
        num_candidates = min(5, max(2, int(duration / 3)))
        candidate_times = []
        for k in range(num_candidates):
            t = start + duration * (k + 1) / (num_candidates + 1)
            t = min(t, video.duration - 0.1)
            candidate_times.append(t)
        
        # 取得所有候選幀並計算視覺豐富度 (用像素標準差)
        best_time = candidate_times[0]
        best_score = -1
        
        for t in candidate_times:
            frame = video.get_frame(t)
            # 計算像素標準差 — 越高代表畫面細節越豐富，越不像純色/轉場
            import numpy as np
            score = float(np.std(frame))
            if score > best_score:
                best_score = score
                best_time = t
        
        screenshot_filename = f"key_{i:03d}.jpg"
        screenshot_path = screenshot_dir / screenshot_filename
        video.save_frame(str(screenshot_path), t=best_time)
        screenshot_paths.append(screenshot_filename)
        print(f"  [{i+1}/{len(segments)}] {seg.get('title', '')} -> {best_time:.1f}s (score: {best_score:.1f})")
        
    video.close()
    return screenshot_paths

def generate_html(video_name: str, summary: str, segments: list, screenshot_paths: list, output_html_path: Path):
    """
    生成包含摘要、關鍵知識點與截圖的 HTML 報告
    """
    print(f"正在生成優化後的 HTML 報告: {output_html_path.name}...")
    
    rows_html = ""
    for i, (seg, img_name) in enumerate(zip(segments, screenshot_paths)):
        safe_img_path = f"screenshots/{video_name.replace(' ', '_')}/{img_name}"
        title = seg.get('title', f'知識點 {i+1}')
        rows_html += f"""
        <div class="segment">
            <div class="segment-image">
                <img src="{safe_img_path}" alt="{title}">
            </div>
            <div class="segment-content">
                <div class="segment-title">{title}</div>
                <div class="timestamp">{seg['start']:.1f}s - {seg['end']:.1f}s</div>
                <div class="segment-text">
                    {seg['text']}
                </div>
            </div>
        </div>
        """
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>影片知識點報告 - {video_name}</title>
    <style>
        body {{ font-family: 'Noto Sans TC', sans-serif, 'Segoe UI'; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 30px; background-color: #f8f9fa; }}
        h1 {{ color: #1a2a6c; text-align: center; margin-bottom: 30px; font-size: 2.2em; }}
        .summary-box {{ background: #ffffff; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 40px; border-top: 6px solid #b21f1f; }}
        .summary-title {{ font-weight: bold; font-size: 1.4em; margin-bottom: 15px; color: #b21f1f; display: flex; align-items: center; }}
        .summary-title::before {{ content: '📝'; margin-right: 10px; }}
        .segment {{ display: flex; background: white; margin-bottom: 30px; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.08); transition: transform 0.2s; }}
        .segment:hover {{ transform: translateY(-3px); }}
        .segment-image {{ flex: 0 0 350px; overflow: hidden; border-right: 1px solid #eee; }}
        .segment-image img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
        .segment-content {{ padding: 20px; flex: 1; display: flex; flex-direction: column; justify-content: start; }}
        .segment-title {{ font-size: 1.25em; font-weight: bold; color: #1a2a6c; margin-bottom: 6px; }}
        .timestamp {{ color: #888; font-size: 0.82em; margin-bottom: 12px; }}
        .segment-text {{ font-size: 1.05em; line-height: 1.7; color: #444; }}
        @media (max-width: 768px) {{
            .segment {{ flex-direction: column; }}
            .segment-image {{ flex: 0 0 auto; }}
        }}
    </style>
</head>
<body>
    <h1>影片知識點詳細報告</h1>
    <div style="text-align: center; margin-bottom: 20px; color: #666;">
        <strong>檔名:</strong> {video_name}
    </div>
    
    <div class="summary-box">
        <div class="summary-title">內容要點總結</div>
        <div style="font-size: 1.1em;">{summary}</div>
    </div>

    <div class="segments-container">
        {rows_html}
    </div>
    
    <footer style="text-align: center; padding: 40px; color: #888; font-size: 0.9em;">
        Generated by Mistral AI Video Analyzer
    </footer>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("錯誤：請在 .env 檔案中設定 MISTRAL_API_KEY")
        return

    client = Mistral(api_key=api_key)
    
    video_dir = Path("Video")
    output_base_dir = Path("output")
    temp_dir = Path("temp_audio")
    
    output_base_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    
    video_extensions = [".mp4", ".mkv", ".mov", ".avi"]
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]
    
    if not videos:
        print(f"在 {video_dir} 目錄下找不到影片檔案。")
        return

    for video_path in videos:
        print(f"\n--- 開始處理影片: {video_path.name} ---")
        
        json_file = output_base_dir / f"{video_path.stem}_transcription.json"
        
        # 快取機制：如果 JSON 已存在，跳過轉錄與翻譯，只重新生成截圖與 HTML
        if json_file.exists():
            print(f"偵測到已有快取 JSON: {json_file.name}")
            print("跳過轉錄與翻譯，直接使用快取資料重新生成截圖與 HTML...")
            with open(json_file, "r", encoding="utf-8") as f:
                processed_data = json.load(f)
        else:
            # 1. 提取音訊
            audio_path = temp_dir / f"{video_path.stem}.mp3"
            extract_audio(video_path, audio_path)
            
            try:
                # 2. 轉錄
                transcription = transcribe_with_mistral(client, audio_path)
                
                # 3. 翻譯與篩選重點 (JSON)
                processed_data = process_and_summarize(client, transcription, video_path.name)
                
                # 儲存 JSON（作為快取）
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                print(f"JSON 已儲存（作為快取）: {json_file}")
                
            except Exception as e:
                print(f"處理影片時發生錯誤: {e}")
                import traceback
                traceback.print_exc()
                continue
            finally:
                if audio_path.exists():
                    audio_path.unlink()
        
        try:
            # 4. 自動截圖 (僅限關鍵知識點)
            safe_video_name = video_path.name.replace(' ', '_')
            screenshot_dir = output_base_dir / "screenshots" / safe_video_name
            screenshot_paths = capture_screenshots(video_path, processed_data["key_moments"], screenshot_dir)
            
            # 5. 生成 HTML 報告
            html_file = output_base_dir / f"{video_path.stem}_report_v2.html"
            generate_html(
                video_path.name, 
                processed_data["summary"], 
                processed_data["key_moments"], 
                screenshot_paths, 
                html_file
            )
            
            print(f"完成！")
            print(f"JSON 結果: {json_file}")
            print(f"HTML 報告: {html_file}")
            
        except Exception as e:
            print(f"生成截圖或 HTML 時發生錯誤: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
