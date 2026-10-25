let distributionChart = null;
let datasetRows = [];
const $ = (id) => document.getElementById(id);
function showAlert(message, error=false) {
  const el = $("globalAlert"); el.textContent = message; el.classList.remove("hidden","error");
  if (error) el.classList.add("error");
  window.scrollTo({top:0,behavior:"smooth"});
}
function clearAlert(){ $("globalAlert").classList.add("hidden"); }
function percent(x){ return x == null ? "—" : (Number(x)*100).toFixed(1)+"%"; }
function goPage(name){
  document.querySelectorAll(".page-section").forEach(s=>s.classList.toggle("hidden",s.id!==name));
  document.querySelectorAll(".nav-link").forEach(b=>b.classList.toggle("active",b.dataset.target===name));
  const titles={dashboard:"Dashboard",predict:"Prediksi Status Gizi",dataset:"Dataset Balita",evaluation:"Evaluasi Model",history:"Riwayat Prediksi"};
  $("pageTitle").textContent=titles[name]||"Dashboard";
  if(name==="history") loadHistory();
  if(name==="dataset") loadDataset();
  window.scrollTo({top:0,behavior:"smooth"});
}
document.querySelectorAll("[data-target]").forEach(b=>b.addEventListener("click",()=>goPage(b.dataset.target)));
document.querySelectorAll("[data-go]").forEach(b=>b.addEventListener("click",()=>goPage(b.dataset.go)));

async function api(url, options={}){
  const res=await fetch(url,options); const data=await res.json();
  if(!res.ok) throw new Error(data.error||"Terjadi kesalahan.");
  return data;
}
async function refreshStatus(){
  try{
    const d=await api("/api/status");
    if(!d.ready){
      $("sidebarStatus").textContent="Belum dilatih"; $("topStatus").textContent="Menunggu dataset";
      $("predictBtn").disabled=true; return;
    }
    $("sidebarStatus").textContent="Model siap"; $("topStatus").textContent="Model aktif";
    document.querySelector(".live-pill i").style.background="#20b49c";
    $("predictBtn").disabled=false;
    $("statRows").textContent=d.rows.toLocaleString("id-ID");
    $("statClasses").textContent=d.classes;
    $("statAccuracy").textContent=percent(d.accuracy);
    $("statK").textContent=d.best_k;
    $("trainAcc").textContent=percent(d.train_accuracy);
    $("testAcc").textContent=percent(d.accuracy);
    $("testCount").textContent=d.test_size;
    $("modelReadyText").textContent=`${d.source_name} · ${d.usable_rows} baris berlabel`;
    $("modelState").textContent="Siap"; $("modelState").className="mini-state good";
    $("modelBullet").textContent="✓";
    $("datasetInfo").textContent=`File aktif: ${d.source_name} · ${d.rows} baris · ${d.missing} nilai kosong terdeteksi · ${d.duplicates} baris duplikat.`;
    drawDistribution(d.class_counts); drawConfusion(d.labels,d.confusion); drawReport(d.labels,d.report);
    await loadDataset();
  }catch(e){showAlert(e.message,true);}
}
function drawDistribution(counts){
  $("emptyDistribution").style.display="none";
  const ctx=$("distributionChart");
  if(distributionChart) distributionChart.destroy();
  distributionChart=new Chart(ctx,{type:"doughnut",data:{labels:Object.keys(counts),datasets:[{data:Object.values(counts),backgroundColor:["#21b6a4","#3f8ed0","#8c80d9","#f0b45e","#eb8c8c","#73b8c2"],borderWidth:0,hoverOffset:5}]},options:{responsive:true,maintainAspectRatio:false,cutout:"66%",plugins:{legend:{position:"bottom",labels:{usePointStyle:true,boxWidth:7,padding:16,font:{family:"DM Sans",size:10}}}}}});
}
function drawConfusion(labels,cm){
  if(!labels||!cm){$("confusionWrap").innerHTML='<p class="muted">Belum ada hasil evaluasi.</p>';return;}
  let html='<table class="confusion-table"><thead><tr><th>Aktual \\ Prediksi</th>'+labels.map(l=>`<th>${esc(l)}</th>`).join("")+'</tr></thead><tbody>';
  cm.forEach((row,i)=>{html+=`<tr><th>${esc(labels[i])}</th>`+row.map(v=>`<td>${v}</td>`).join("")+'</tr>';});
  $("confusionWrap").innerHTML=html+'</tbody></table>';
}
function drawReport(labels,report){
  let html="";
  (labels||[]).forEach(label=>{const r=report[label]||{};html+=`<tr><td>${esc(label)}</td><td>${percent(r.precision)}</td><td>${percent(r.recall)}</td><td>${percent(r["f1-score"])}</td><td>${r.support??0}</td></tr>`;});
  $("reportBody").innerHTML=html||'<tr><td colspan="5" class="empty-cell">Belum ada hasil evaluasi.</td></tr>';
}
function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
$("predictForm").addEventListener("submit",async e=>{
  e.preventDefault(); clearAlert();
  const btn=$("predictBtn"); btn.disabled=true; btn.textContent="Memproses...";
  try{
    const d=await api("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({age:$("age").value,gender:$("gender").value,height:$("height").value})});
    $("resultPlaceholder").classList.add("hidden"); $("resultContent").classList.remove("hidden");
    $("predictionValue").textContent=d.prediction; goPage("predict"); await loadHistory();
  }catch(err){showAlert(err.message,true);}
  finally{btn.disabled=false;btn.innerHTML='Prediksi status gizi <span>→</span>'; await refreshPredictReady();}
});
async function refreshPredictReady(){try{const d=await api("/api/status");$("predictBtn").disabled=!d.ready;}catch(e){$("predictBtn").disabled=true;}}
$("uploadForm").addEventListener("submit",async e=>{
  e.preventDefault();clearAlert();
  const file=$("csvFile").files[0]; if(!file){showAlert("Pilih file CSV terlebih dahulu.",true);return;}
  const btn=e.submitter; const old=btn.textContent;btn.disabled=true;btn.textContent="Melatih model...";
  const fd=new FormData();fd.append("file",file);
  try{
    const d=await api("/api/upload",{method:"POST",body:fd});
    showAlert(d.message+" File aktif: "+d.source_name); $("csvFile").value=""; await refreshStatus(); goPage("dataset");
  }catch(err){showAlert(err.message,true);}
  finally{btn.disabled=false;btn.textContent=old;}
});
async function loadDataset(){
  try{
    const d=await api("/api/dataset");datasetRows=d.rows||[];renderDataset(datasetRows);
    $("tableSub").textContent=`Menampilkan ${datasetRows.length} baris dari ${d.total||0} total (maksimal 500).`;
  }catch(e){}
}
function renderDataset(rows){
  const body=$("datasetBody");
  if(!rows.length){body.innerHTML='<tr><td colspan="4" class="empty-cell">Unggah dataset CSV untuk melihat data.</td></tr>';return;}
  body.innerHTML=rows.map(r=>`<tr><td>${esc(r["Umur (bulan)"]??"—")}</td><td>${esc(r["Jenis Kelamin"]??"—")}</td><td>${esc(r["Tinggi Badan (cm)"]??"—")}</td><td>${esc(r["Status Gizi"]??"—")}</td></tr>`).join("");
}
$("tableSearch").addEventListener("input",()=>{
  const q=$("tableSearch").value.toLowerCase().trim();
  renderDataset(datasetRows.filter(r=>Object.values(r).some(v=>String(v??"").toLowerCase().includes(q))));
});
async function loadHistory(){
  try{const d=await api("/api/history");const rows=d.rows||[];
    $("historyBody").innerHTML=rows.length?rows.map(r=>`<tr><td>${esc(r.waktu)}</td><td>${esc(r.umur_bulan)} bulan</td><td>${esc(r.jenis_kelamin)}</td><td>${esc(r.tinggi_cm)} cm</td><td><b>${esc(r.prediksi)}</b></td></tr>`).join(""):'<tr><td colspan="5" class="empty-cell">Belum ada prediksi.</td></tr>';
  }catch(e){}
}
$("refreshHistory").addEventListener("click",loadHistory);
refreshStatus();
