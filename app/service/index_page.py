"""Self-contained HTML chat UI served at GET / (no external assets/CDN)."""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>System Design RAG</title>
<style>
  :root{--bg:#0f172a;--card:#1e293b;--accent:#38bdf8;--text:#e2e8f0;--muted:#94a3b8;}
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text);}
  .wrap{max-width:820px;margin:0 auto;padding:28px 24px;}
  h1{font-size:20px;margin:0 0 4px;}
  .sub{color:var(--muted);font-size:13px;margin-bottom:20px;}
  form{display:flex;gap:8px;}
  input[type=text]{flex:1;padding:12px 14px;border-radius:10px;border:1px solid #334155;
    background:var(--card);color:var(--text);font-size:15px;}
  button{padding:12px 18px;border:0;border-radius:10px;background:var(--accent);color:#0b1220;
    font-weight:600;cursor:pointer;font-size:15px;}
  button:disabled{opacity:.5;cursor:not-allowed;}
  .ex{margin-top:14px;font-size:12px;color:var(--muted);}
  .ex a{color:var(--accent);cursor:pointer;text-decoration:underline;margin-right:12px;}
  .answer{margin-top:20px;background:var(--card);border:1px solid #334155;border-radius:12px;
    padding:18px;white-space:pre-wrap;line-height:1.55;min-height:44px;}
  .meta{margin-top:10px;font-size:12px;color:var(--muted);}
  .sources{margin-top:8px;font-size:12px;color:var(--muted);word-break:break-all;}
  .sources span{color:var(--accent);}
  .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;
    background:#064e3b;color:#6ee7b7;}
  .badge.no{background:#7f1d1d;color:#fecaca;}
</style>
</head>
<body>
<div class="wrap">
  <h1>System Design RAG</h1>
  <div class="sub">Ask anything about system design &mdash; answers are grounded in the ingested
    tutorials and cite their sources.</div>
  <form id="f">
    <input id="q" type="text" autocomplete="off"
      placeholder="e.g. What are the trade-offs of sharding a database?"/>
    <button id="b" type="submit">Ask</button>
  </form>
  <div class="ex">Try:
    <a data-q="What is the CAP theorem?">CAP theorem</a>
    <a data-q="How does consistent hashing work?">consistent hashing</a>
    <a data-q="When should I use a message queue?">message queues</a>
    <a data-q="Compare SQL and NoSQL databases.">SQL vs NoSQL</a>
  </div>
  <div id="ans" class="answer">Your answer will appear here.</div>
  <div id="meta" class="meta"></div>
  <div id="src" class="sources"></div>
</div>
<script>
const f=document.getElementById('f'),q=document.getElementById('q'),b=document.getElementById('b'),
  ans=document.getElementById('ans'),meta=document.getElementById('meta'),src=document.getElementById('src');
document.querySelectorAll('.ex a').forEach(a=>a.onclick=()=>{q.value=a.dataset.q;f.requestSubmit();});
f.onsubmit=async(e)=>{
  e.preventDefault();
  const question=q.value.trim(); if(!question) return;
  b.disabled=true; ans.textContent='Thinking\\u2026'; meta.textContent=''; src.textContent='';
  try{
    const r=await fetch('/query',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({question})});
    if(!r.ok){ans.textContent='Error '+r.status+': '+(await r.text()); return;}
    const d=await r.json();
    ans.textContent=d.answer;
    const badge=d.grounded?'<span class=\\"badge\\">grounded</span>'
      :'<span class=\\"badge no\\">low grounding</span>';
    meta.innerHTML=badge+' &nbsp; '+d.model+' &nbsp; '+d.latency_ms+' ms';
    if(d.sources&&d.sources.length){
      src.innerHTML='Sources: '+d.sources.map(s=>'<span>'+s.source+'/'+s.path+'</span>').join(' &middot; ');
    }
  }catch(err){ans.textContent='Request failed: '+err;}
  finally{b.disabled=false;}
};
</script>
</body>
</html>
"""
