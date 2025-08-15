/**
 * 返回顶部按钮功能
 * 自动显示/隐藏按钮，并提供平滑滚动到顶部的功能
 */

(function() {
    'use strict';
    
    // 等待DOM加载完成
    document.addEventListener('DOMContentLoaded', function() {
        // 创建返回顶部按钮
        createBackToTopButton();
        
        // 监听滚动事件
        window.addEventListener('scroll', toggleButtonVisibility);
        
        // 监听页面重新加载
        window.addEventListener('load', toggleButtonVisibility);
    });
    
    /**
     * 创建返回顶部按钮
     */
    function createBackToTopButton() {
        // 检查按钮是否已存在
        if (document.getElementById('back-to-top')) {
            return;
        }
        
        // 创建按钮元素
        const button = document.createElement('button');
        button.id = 'back-to-top';
        button.setAttribute('title', '返回顶部');
        button.setAttribute('aria-label', '返回顶部');
        
        // 添加点击事件
        button.addEventListener('click', scrollToTop);
        
        // 将按钮添加到页面
        document.body.appendChild(button);
    }
    
    /**
     * 根据滚动位置显示/隐藏按钮
     */
    function toggleButtonVisibility() {
        const button = document.getElementById('back-to-top');
        if (!button) return;
        
        // 当页面滚动超过300px时显示按钮
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        if (scrollTop > 300) {
            button.classList.add('show');
        } else {
            button.classList.remove('show');
        }
    }
    
    /**
     * 平滑滚动到页面顶部
     */
    function scrollToTop() {
        // 使用现代浏览器的平滑滚动
        if ('scrollBehavior' in document.documentElement.style) {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        } else {
            // 对于不支持平滑滚动的浏览器，使用渐进式滚动
            const scrollStep = -window.scrollY / (500 / 15);
            const scrollInterval = setInterval(function() {
                if (window.scrollY !== 0) {
                    window.scrollBy(0, scrollStep);
                } else {
                    clearInterval(scrollInterval);
                }
            }, 15);
        }
    }
    
    /**
     * 添加键盘支持（可选）
     */
    document.addEventListener('keydown', function(event) {
        // 按下 Home 键时滚动到顶部
        if (event.key === 'Home' && event.ctrlKey) {
            event.preventDefault();
            scrollToTop();
        }
    });
})();