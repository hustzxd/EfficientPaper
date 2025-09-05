import subprocess
import html
import re

def define_env(env):
    """
    Hook function for mkdocs-macros-plugin
    """

    GITHUB_REPO = "https://github.com/hustzxd/EfficientPaper"  # 换成你的仓库地址
    
# 调试信息已移除

    @env.macro
    def latest_commits(*args, **kwargs):
        """
        获取指定目录的最近 n 次提交，并生成带 GitHub 链接的表格
        支持多种调用方式:
        1. latest_commits("docs/weekly_paper", 5)
        2. latest_commits("docs/weekly_paper, meta", 5) 
        3. latest_commits("docs/weekly_paper", "meta", 5)
        4. latest_commits("docs/weekly_paper", "meta", n=5)
        """
        # 解析参数
        n = kwargs.get('n', 3)  # 默认显示3个提交
        paths = []
        
        # 处理位置参数
        for arg in args:
            if isinstance(arg, int):
                n = arg  # 如果是数字，当作n处理
            elif isinstance(arg, str):
                paths.append(arg)
        
        # 如果没有提供路径，使用默认值
        if not paths:
            paths = ["docs"]
        
        # 将多个路径合并成一个字符串（兼容原有逻辑）
        if len(paths) == 1 and "," in paths[0]:
            # 情况：latest_commits("docs/weekly_paper, meta", 5)
            path = paths[0]
        else:
            # 情况：latest_commits("docs/weekly_paper", "meta", 5)
            path = ", ".join(paths)
        
        # 如果path包含逗号，则为多个目录
        if "," in path:
            paths = [p.strip() for p in path.split(",")]
            
            # 为每个路径获取提交记录，然后合并
            all_commits = []
            separator = "|||"
            pretty = f"--pretty=format:%h{separator}%ad{separator}%an{separator}%s{separator}%ai"
            
            for single_path in paths:
                cmd = [
                    "git", "log",
                    f"-{n*2}",  # 获取更多提交以便后续排序
                    pretty,
                    "--date=short",
                    "--", single_path
                ]
                try:
                    output = subprocess.check_output(cmd, encoding='utf-8').strip()
                    if output:
                        for line in output.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split(separator)
                            if len(parts) == 5:
                                commit, date, author, msg, iso_date = [x.strip() for x in parts]
                                all_commits.append({
                                    'commit': commit,
                                    'date': date,
                                    'author': author,
                                    'msg': msg,
                                    'iso_date': iso_date,
                                    'path': single_path
                                })
                except subprocess.CalledProcessError:
                    continue
            
            # 按日期排序并取前n个
            all_commits.sort(key=lambda x: x['iso_date'], reverse=True)
            all_commits = all_commits[:n]
            
            if not all_commits:
                return "<p><em>No commits found for any of the specified paths</em></p>"
                
            rows = []
            for commit_info in all_commits:
                commit = html.escape(commit_info['commit'])
                date = html.escape(commit_info['date'])
                author = html.escape(commit_info['author'])
                msg = html.escape(commit_info['msg'])
                path_name = html.escape(commit_info['path'])
                
                # Commit 链接
                commit_link = f'<a href="{GITHUB_REPO}/commit/{commit}" target="_blank">{commit}</a>'
                
                # 把 message 里的 #123 转成 issue 链接
                msg = re.sub(
                    r"#(\d+)",
                    lambda m: f'<a href="{GITHUB_REPO}/issues/{m.group(1)}" target="_blank">#{m.group(1)}</a>',
                    msg
                )
                
                rows.append(
                    f'<tr><td style="border: 1px solid #ddd; padding: 8px;">{commit_link}</td>'
                    f'<td style="border: 1px solid #ddd; padding: 8px;">{date}</td>'
                    f'<td style="border: 1px solid #ddd; padding: 8px;">{author}</td>'
                    f'<td style="border: 1px solid #ddd; padding: 8px;">{msg}</td>'
                    f'<td style="border: 1px solid #ddd; padding: 8px;"><code>{path_name}</code></td></tr>'
                )
            
            table = f"""
<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
    <thead>
        <tr style="background-color: #f5f5f5;">
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Commit</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Date</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Author</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Message</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Path</th>
        </tr>
    </thead>
    <tbody>
        {''.join(rows)}
    </tbody>
</table>"""
            
            return table
        
        else:
            # 单个目录的原有逻辑
            separator = "|||"
            pretty = f"--pretty=format:%h{separator}%ad{separator}%an{separator}%s"

            cmd = [
                "git", "log",
                f"-{n}",
                pretty,
                "--date=short",
                "--", path
            ]
            try:
                output = subprocess.check_output(cmd, encoding='utf-8').strip()
            except subprocess.CalledProcessError:
                return "<p><em>No git history available</em></p>"

            if not output:
                return "<p><em>No commits found for this path</em></p>"

            rows = []
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(separator)
                if len(parts) != 4:
                    # 调试信息：如果解析失败，显示原始行
                    env.variables.setdefault('debug_lines', []).append(f"Parse failed: {line}")
                    continue

                commit, date, author, msg = [html.escape(x.strip()) for x in parts]

                # Commit 链接
                commit_link = f'<a href="{GITHUB_REPO}/commit/{commit}" target="_blank">{commit}</a>'

                # 把 message 里的 #123 转成 issue 链接
                msg = re.sub(
                    r"#(\d+)",
                    lambda m: f'<a href="{GITHUB_REPO}/issues/{m.group(1)}" target="_blank">#{m.group(1)}</a>',
                    msg
                )

                rows.append(
                    f'<tr><td style="border: 1px solid #ddd; padding: 8px;">{commit_link}</td>'
                    f'<td style="border: 1px solid #ddd; padding: 8px;">{date}</td>'
                    f'<td style="border: 1px solid #ddd; padding: 8px;">{author}</td>'
                    f'<td style="border: 1px solid #ddd; padding: 8px;">{msg}</td></tr>'
                )

            if not rows:
                return "<p><em>No valid commits found</em></p>"

            table = f"""
<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
    <thead>
        <tr style="background-color: #f5f5f5;">
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Commit</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Date</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Author</th>
            <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Message</th>
        </tr>
    </thead>
    <tbody>
        {''.join(rows)}
    </tbody>
</table>"""

            return table
    
    @env.macro
    def latest_commits_from_paths(paths, n=3):
        """
        从多个路径获取最近的 n 次提交，并生成带 GitHub 链接的表格
        paths: 路径列表，如 ["docs/weekly_paper", "meta"] 或 "docs/weekly_paper, meta"
        """
        if isinstance(paths, str):
            # 如果传入的是字符串，按逗号分割
            paths = [p.strip() for p in paths.split(",")]
        
        # 使用existing logic，转换成逗号分隔的字符串
        path_str = ", ".join(paths)
        return latest_commits(path_str, n)
    
    @env.macro
    def latest_commits_multi(*paths, n=3):
        """
        从多个路径获取最近的 n 次提交（支持多个参数）
        用法: latest_commits_multi("docs/weekly_paper", "meta", n=5)
        """
        if len(paths) == 0:
            return "<p><em>No paths provided</em></p>"
        
        # 使用existing logic，转换成逗号分隔的字符串
        path_str = ", ".join(paths)
        return latest_commits(path_str, n)
