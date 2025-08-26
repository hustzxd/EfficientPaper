# Contributors

<div id="contributors-count" style="font-weight:bold; margin-bottom:15px;"></div>
<div id="contributors"></div>

<!-- Modal -->
<div id="contributor-modal" style="
    display:none;
    position:fixed;
    top:0; left:0;
    width:100%; height:100%;
    background:rgba(0,0,0,0.6);
    justify-content:center;
    align-items:center;
    z-index:1000;
">
  <div style="
      background:#fff;
      padding:25px;
      border-radius:12px;
      text-align:center;
      max-width:320px;
      position:relative;
      box-shadow: 0 10px 30px rgba(0,0,0,0.3);
  ">
    <span id="modal-close" style="
        position:absolute;
        top:8px;
        right:12px;
        cursor:pointer;
        font-weight:bold;
        font-size:20px;
    ">×</span>
    <img id="modal-avatar" src="" width="130" height="130" style="border-radius:50%; border:2px solid #ddd;">
    <div id="modal-login" style="font-size:20px; margin-top:10px; font-weight:bold;"></div>
    <div id="modal-commits" style="font-size:14px; color:#666; margin-top:5px;"></div>
    <a id="modal-link" href="#" target="_blank" style="display:block; margin-top:10px; color:#4caf50; font-weight:bold;">View GitHub Profile</a>
  </div>
</div>

<style>
  :root {
    --progress-color: #4caf50; /* 可调主题颜色 */
  }

  .contributor {
    text-align: center;
    transition: transform 0.2s;
    cursor: pointer;
    position: relative;
    border-radius: 12px;
    padding: 8px;
  }

  .contributor:hover {
    transform: scale(1.08);
  }

  .contributor img {
    border-radius: 50%;
    display: block;
    margin: 0 auto;
    transition: transform 0.2s;
  }

  .tooltip {
    visibility: hidden;
    background-color: rgba(0,0,0,0.8);
    color: #fff;
    text-align: center;
    border-radius: 5px;
    padding: 4px 8px;
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
    font-size: 12px;
    opacity: 0;
    transition: opacity 0.2s;
    z-index: 1;
  }

  .contributor:hover .tooltip {
    visibility: visible;
    opacity: 1;
  }

  /* 网格布局响应式 */
  #contributors > div {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 20px;
    text-align: center;
  }

  /* 进度条样式 */
  .progress-bar {
    background:#eee;
    border-radius:5px;
    overflow:hidden;
    margin-top:4px;
    height:6px;
  }

  .progress-bar-fill {
    height:100%;
    background: var(--progress-color);
  }
</style>

<script src="https://unpkg.com/axios/dist/axios.min.js"></script>
<script>
axios.get("https://api.github.com/repos/hustzxd/EfficientPaper/contributors")
  .then(function (res) {
    let contributors = res.data.sort((a,b) => b.contributions - a.contributions);
    let totalCommits = contributors.reduce((sum,c) => sum + c.contributions, 0);
    document.getElementById("contributors-count").innerText = "Total Contributors: " + contributors.length;

    const medalColors = ["#FFD700", "#C0C0C0", "#CD7F32"]; // Top3 边框颜色

    let html = '<div>';
    contributors.forEach((c,index)=>{
      let percent = ((c.contributions/totalCommits)*100).toFixed(1);
      let borderColor = index<3? medalColors[index]:"#ddd";
      let shadow = index<3? "0 4px 15px rgba(0,0,0,0.2)":"none";

      html += `
        <div class="contributor" 
             data-login="${c.login}" 
             data-avatar="${c.avatar_url}" 
             data-contrib="${c.contributions}" 
             data-url="${c.html_url}" 
             style="border:3px solid ${borderColor}; box-shadow:${shadow};">
          <img src="${c.avatar_url}" width="70" height="70">
          <div style="font-size:13px; margin-top:6px;">${c.login}</div>
          <div style="font-size:11px; color:#666; margin-top:4px;">${c.contributions} commits (${percent}%)</div>
          <div class="progress-bar">
            <div class="progress-bar-fill" style="width:${percent}%"></div>
          </div>
          <div class="tooltip">${c.login} - ${c.contributions} commits</div>
        </div>
      `;
    });
    html += '</div>';
    document.getElementById("contributors").innerHTML = html;

    // 点击显示 Modal
    document.querySelectorAll('.contributor').forEach(el=>{
      el.addEventListener('click', ()=>{
        document.getElementById('modal-avatar').src = el.dataset.avatar;
        document.getElementById('modal-login').innerText = el.dataset.login;
        document.getElementById('modal-commits').innerText = el.dataset.contrib + ' commits';
        document.getElementById('modal-link').href = el.dataset.url;
        document.getElementById('contributor-modal').style.display = 'flex';
      });
    });

    // 关闭 Modal
    document.getElementById('modal-close').addEventListener('click', ()=>{
      document.getElementById('contributor-modal').style.display = 'none';
    });
    document.getElementById('contributor-modal').addEventListener('click',(e)=>{
      if(e.target.id==='contributor-modal') document.getElementById('contributor-modal').style.display='none';
    });
  });
</script>
