document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll('a[href$=".prototxt"]').forEach(function (link) {
        link.addEventListener("click", function (e) {
            e.preventDefault();

            fetch(link.getAttribute("href"))
                .then(response => response.text())
                .then(text => {
                    // 背景遮罩
                    const overlay = document.createElement("div");
                    overlay.style.position = "fixed";
                    overlay.style.top = "0";
                    overlay.style.left = "0";
                    overlay.style.width = "100%";
                    overlay.style.height = "100%";
                    overlay.style.background = "rgba(0,0,0,0.5)";
                    overlay.style.zIndex = "9998";
                    overlay.style.opacity = "0";
                    overlay.style.transition = "opacity 0.3s ease";

                    // 弹窗
                    const modal = document.createElement("div");
                    modal.style.position = "fixed";
                    modal.style.top = "50%";
                    modal.style.left = "50%";
                    modal.style.transform = "translate(-50%, -50%) scale(0.8)";
                    modal.style.width = "80%";
                    modal.style.maxWidth = "900px";
                    modal.style.maxHeight = "80%";
                    modal.style.background = "#faf9f7";
                    modal.style.borderRadius = "10px";
                    modal.style.boxShadow = "0 4px 20px rgba(0,0,0,0.3)";
                    modal.style.overflow = "hidden";
                    modal.style.display = "flex";
                    modal.style.flexDirection = "column";
                    modal.style.zIndex = "9999";
                    modal.style.opacity = "0";
                    modal.style.transition = "all 0.3s ease";

                    // 头部
                    const header = document.createElement("div");
                    header.style.background = "#2c3e50";
                    header.style.color = "white";
                    header.style.padding = "10px 15px";
                    header.style.fontSize = "16px";
                    header.style.fontWeight = "bold";
                    header.style.display = "flex";
                    header.style.justifyContent = "space-between";
                    header.style.alignItems = "center";
                    header.innerHTML = `
                        <span>${link.getAttribute("href")}</span>
                        <button id="closeModal" style="
                            background: none;
                            border: none;
                            color: white;
                            font-size: 18px;
                            cursor: pointer;
                        ">&times;</button>
                    `;

                    // 内容区域
                    const content = document.createElement("pre");
                    content.style.flex = "1";
                    content.style.margin = "0";
                    content.style.padding = "15px";
                    content.style.background = "#1e1e1e";
                    content.style.overflow = "auto";
                    content.style.borderRadius = "0 0 10px 10px";

                    const code = document.createElement("code");
                    code.className = "language-prototxt";
                    code.textContent = text;
                    content.appendChild(code);

                    modal.appendChild(header);
                    modal.appendChild(content);

                    // 点击遮罩或关闭按钮时移除
                    const closeModal = () => {
                        overlay.style.opacity = "0";
                        modal.style.opacity = "0";
                        modal.style.transform = "translate(-50%, -50%) scale(0.8)";
                        setTimeout(() => {
                            overlay.remove();
                            modal.remove();
                        }, 300);
                    };

                    overlay.addEventListener("click", closeModal);
                    header.querySelector("#closeModal").addEventListener("click", closeModal);

                    document.body.appendChild(overlay);
                    document.body.appendChild(modal);

                    // 动画触发
                    setTimeout(() => {
                        overlay.style.opacity = "1";
                        modal.style.opacity = "1";
                        modal.style.transform = "translate(-50%, -50%) scale(1)";
                    }, 10);

                    // 代码高亮
                    setTimeout(() => {
                        if (typeof Prism !== "undefined") {
                            Prism.highlightElement(code);
                        } else if (typeof hljs !== "undefined") {
                            hljs.highlightElement(code);
                        }
                    }, 50);
                });
        });
    });
});
