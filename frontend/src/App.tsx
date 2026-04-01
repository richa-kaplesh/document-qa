import { useState, useRef } from "react"
import Eval from "./Eval"

const API_URL = "http://localhost:8000"

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@400;500;600&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f5f4f0; min-height: 100vh; font-family: 'DM Sans', sans-serif; }
  .wrap { min-height: 100vh; display: flex; align-items: flex-start; justify-content: center; padding: 60px 24px; }
  .card { width: 100%; max-width: 680px; }
  .badge { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 500; color: #999; letter-spacing: 0.04em; margin-bottom: 24px; }
  .badge-dot { width: 6px; height: 6px; border-radius: 50%; background: #6366f1; }
  h1 { font-family: 'DM Serif Display', serif; font-size: clamp(36px, 7vw, 56px); color: #1a1a1a; line-height: 1.05; letter-spacing: -0.02em; margin-bottom: 12px; }
  h1 em { font-style: italic; color: #6366f1; }
  .sub { font-size: 15px; color: #999; line-height: 1.6; margin-bottom: 40px; }
  hr { border: none; border-top: 1px solid #e0ddd8; margin-bottom: 32px; }
  .tabs { display: flex; gap: 0; margin-bottom: 32px; border-bottom: 1.5px solid #e0ddd8; }
  .tab { font-size: 14px; padding: 10px 20px; background: none; border: none; cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.2s; color: #999; font-family: 'DM Sans', sans-serif; }
  .tab.active { border-bottom-color: #1a1a1a; color: #1a1a1a; font-weight: 500; }
  .upload-zone { border: 2px dashed #e0ddd8; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.2s; margin-bottom: 24px; }
  .upload-zone:hover { border-color: #6366f1; background: #f8f8ff; }
  .upload-zone.uploaded { border-color: #22c55e; background: #f0faf4; border-style: solid; }
  .upload-icon { font-size: 32px; margin-bottom: 12px; }
  .upload-text { font-size: 15px; color: #666; }
  .upload-sub { font-size: 12px; color: #aaa; margin-top: 6px; }
  .input-group { display: flex; gap: 10px; margin-bottom: 16px; }
  input[type="text"] { flex: 1; padding: 13px 16px; font-family: 'DM Sans', sans-serif; font-size: 15px; color: #1a1a1a; background: #fff; border: 1.5px solid #e0ddd8; border-radius: 8px; outline: none; transition: border-color 0.2s; }
  input[type="text"]:focus { border-color: #1a1a1a; }
  input[type="text"]::placeholder { color: #ccc; }
  button.primary { padding: 13px 24px; font-family: 'DM Sans', sans-serif; font-size: 14px; font-weight: 500; color: #fff; background: #1a1a1a; border: none; border-radius: 8px; cursor: pointer; transition: background 0.2s; white-space: nowrap; }
  button.primary:hover:not(:disabled) { background: #333; }
  button.primary:disabled { opacity: 0.4; cursor: not-allowed; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner { width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.3); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; display: inline-block; margin-right: 8px; }
  @keyframes fade-up { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .answer-box { margin-top: 24px; padding: 24px; background: #fff; border-radius: 12px; border: 1.5px solid #e0ddd8; animation: fade-up 0.3s ease; }
  .answer-label { font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: #6366f1; margin-bottom: 8px; }
  .answer-text { font-size: 15px; color: #1a1a1a; line-height: 1.7; }
  .sources { margin-top: 16px; padding-top: 16px; border-top: 1px solid #f0f0f0; }
  .sources-label { font-size: 11px; color: #aaa; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px; }
  .source-chunk { font-size: 12px; color: #888; background: #f9f9f9; padding: 10px 12px; border-radius: 6px; margin-bottom: 6px; line-height: 1.5; border-left: 3px solid #e0ddd8; }
  .status-bar { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #999; margin-bottom: 20px; padding: 10px 14px; background: #fff; border-radius: 8px; border: 1px solid #f0f0f0; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #e0ddd8; flex-shrink: 0; }
  .status-dot.ready { background: #22c55e; }
`

export default function App() {
  const [activeTab, setActiveTab] = useState<"qa" | "eval">("qa")
  const [uploaded, setUploaded] = useState(false)
  const [docName, setDocName] = useState("")
  const [chunkCount, setChunkCount] = useState(0)
  const [uploading, setUploading] = useState(false)
  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState("")
  const [sourceChunks, setSourceChunks] = useState<string[]>([])
  const [asking, setAsking] = useState(false)
  const [vectorstore, setVectorstore] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    const formData = new FormData()
    formData.append("file", file)
    const res = await fetch(`${API_URL}/upload`, { method: "POST", body: formData })
    const data = await res.json()
    setDocName(data.filename)
    setChunkCount(data.chunks_created)
    setUploaded(true)
    setVectorstore(true)
    setUploading(false)
  }

  async function handleAsk() {
    if (!question.trim() || asking) return
    setAsking(true)
    setAnswer("")
    setSourceChunks([])
    const res = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    })
    const data = await res.json()
    setAnswer(data.answer)
    setSourceChunks(data.source_chunks || [])
    setAsking(false)
  }

  return (
    <>
      <style>{styles}</style>
      <div className="wrap">
        <div className="card">
          <div className="badge"><span className="badge-dot" />RAG System</div>
          <h1>Document <em>Q&A</em></h1>
          <p className="sub">Upload any PDF and ask questions. Powered by Llama 3 + FAISS.</p>
          <hr />

          <div className="tabs">
            <button className={`tab ${activeTab === "qa" ? "active" : ""}`} onClick={() => setActiveTab("qa")}>Ask Questions</button>
            <button className={`tab ${activeTab === "eval" ? "active" : ""}`} onClick={() => setActiveTab("eval")}>Evaluation</button>
          </div>

          {activeTab === "qa" && (
            <>
              <div className="status-bar">
                <span className={`status-dot ${uploaded ? "ready" : ""}`} />
                {uploaded ? `${docName} — ${chunkCount} chunks created` : "No document loaded"}
              </div>

              <div className={`upload-zone ${uploaded ? "uploaded" : ""}`} onClick={() => fileRef.current?.click()}>
                <input ref={fileRef} type="file" accept=".pdf" style={{ display: "none" }} onChange={handleUpload} />
                <div className="upload-icon">{uploaded ? "✓" : "📄"}</div>
                <div className="upload-text">{uploading ? "Processing document..." : uploaded ? `${docName} uploaded` : "Click to upload a PDF"}</div>
                <div className="upload-sub">{uploaded ? `${chunkCount} chunks created` : "PDF files only"}</div>
              </div>

              <div className="input-group">
                <input
                  type="text"
                  placeholder="Ask a question about your document..."
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleAsk()}
                  disabled={!uploaded}
                />
                <button className="primary" onClick={handleAsk} disabled={!uploaded || !question.trim() || asking}>
                  {asking ? <><span className="spinner" />Thinking</> : "Ask"}
                </button>
              </div>

              {answer && (
                <div className="answer-box">
                  <div className="answer-label">Answer</div>
                  <div className="answer-text">{answer}</div>
                  {sourceChunks.length > 0 && (
                    <div className="sources">
                      <div className="sources-label">Source chunks used</div>
                      {sourceChunks.map((chunk, i) => (
                        <div key={i} className="source-chunk">{chunk.slice(0, 200)}...</div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {activeTab === "eval" && <Eval uploaded={uploaded} />}
        </div>
      </div>
    </>
  )
}