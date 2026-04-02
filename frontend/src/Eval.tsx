import { useState } from "react"

const API_URL = "http://localhost:8000"

interface EvalResult {
  question: string
  expected_answer: string
  actual_answer: string
  similarity_score: number
  passed: boolean
}

interface EvalResponse {
  overall_score: number
  passed: number
  total: number
  threshold: number
  results: EvalResult[]
}

const evalStyles = `
  .eval-wrap { padding: 8px 0; }
  .eval-desc { font-size: 14px; color: #888; line-height: 1.6; margin-bottom: 24px; }
  .eval-input { display: flex; gap: 10px; margin-bottom: 24px; }
  .score-box { padding: 24px; background: #fff; border-radius: 12px; border: 1.5px solid #e0ddd8; margin-bottom: 24px; text-align: center; }
  .score-number { font-family: 'DM Serif Display', serif; font-size: 56px; color: #1a1a1a; line-height: 1; }
  .score-label { font-size: 13px; color: #999; margin-top: 8px; }
  .score-sub { font-size: 12px; color: #bbb; margin-top: 4px; }
  .result-item { padding: 16px; background: #fff; border-radius: 10px; border: 1.5px solid #e0ddd8; margin-bottom: 12px; }
  .result-item.passed { border-color: #b6e4c8; }
  .result-item.failed { border-color: #f5c0c0; }
  .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .result-question { font-size: 14px; font-weight: 500; color: #1a1a1a; flex: 1; }
  .result-score { font-size: 13px; font-weight: 500; padding: 3px 10px; border-radius: 20px; }
  .result-score.passed { background: #f0faf4; color: #16a34a; }
  .result-score.failed { background: #fff5f5; color: #dc2626; }
  .result-answers { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .answer-col label { font-size: 10px; color: #aaa; letter-spacing: 0.06em; text-transform: uppercase; display: block; margin-bottom: 4px; }
  .answer-col p { font-size: 13px; color: #555; line-height: 1.5; }
`

export default function Eval({ uploaded }: { uploaded: boolean }) {
  const [datasetPath, setDatasetPath] = useState("../data/golden_dataset.json")
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState<EvalResponse | null>(null)

  async function runEval() {
    if (!uploaded || running) return
    setRunning(true)
    setResults(null)
    const res = await fetch(`${API_URL}/eval`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ golden_dataset_path: datasetPath })
    })
    const data = await res.json()
    setResults(data)
    setRunning(false)
  }

  return (
    <div className="eval-wrap">
      <style>{evalStyles}</style>
      <p className="eval-desc">
        Run the evaluation harness against a golden dataset of known Q&A pairs.
        Each question is scored by comparing the RAG answer to the expected answer
        using cosine similarity. A score above the threshold counts as a pass.
      </p>

      {!uploaded && (
        <p style={{ color: "#dc2626", fontSize: "14px", marginBottom: "16px" }}>
          Upload a document first before running evaluation.
        </p>
      )}

      <div className="eval-input">
        <input
          type="text"
          value={datasetPath}
          onChange={e => setDatasetPath(e.target.value)}
          placeholder="Path to golden_dataset.json"
          disabled={!uploaded}
        />
        <button className="primary" onClick={runEval} disabled={!uploaded || running}>
          {running ? "Running..." : "Run Eval"}
        </button>
      </div>

      {results && (
        <>
          <div className="score-box">
            <div className="score-number">{results.overall_score}%</div>
            <div className="score-label">{results.passed} of {results.total} questions passed</div>
            <div className="score-sub">Similarity threshold: {results.threshold}</div>
          </div>

          {results.results.map((r, i) => (
            <div key={i} className={`result-item ${r.passed ? "passed" : "failed"}`}>
              <div className="result-header">
                <div className="result-question">{r.question}</div>
                <div className={`result-score ${r.passed ? "passed" : "failed"}`}>
                  {r.similarity_score} {r.passed ? "✓" : "✗"}
                </div>
              </div>
              <div className="result-answers">
                <div className="answer-col">
                  <label>Expected</label>
                  <p>{r.expected_answer}</p>
                </div>
                <div className="answer-col">
                  <label>Actual</label>
                  <p>{r.actual_answer}</p>
                </div>
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  )
}