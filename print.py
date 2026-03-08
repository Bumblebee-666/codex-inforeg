import os

# ================= 配置区域 =================

# 输出文件的名称
OUTPUT_FILE = "project_context.txt"

# 需要提取的文件后缀
TARGET_EXTENSIONS = {'.py'}  # 你也可以添加 '.js', '.html', '.css' 等

# 需要忽略的目录 (黑名单)
IGNORE_DIRS = {
    '.git', '__pycache__', 'venv', 'env', '.idea', '.vscode',
    'dist', 'build', 'node_modules', 'migrations'
}

# 需要忽略的具体文件名
IGNORE_FILES = {
    'code_dumper.py',  # 忽略脚本自己
    'package-lock.json',
    'yarn.lock'
}


# ===========================================

def is_ignored(path, names, ignored_set):
    return any(name in ignored_set for name in names)


def collect_code():
    total_files = 0
    project_root = os.getcwd()

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            # 写入项目结构概览（可选，但对AI很有用）
            outfile.write("Project Structure:\n")
            outfile.write("==================\n")

            # 第一遍遍历：生成目录树
            for root, dirs, files in os.walk(project_root):
                # 修改 dirs 列表以跳过忽略的目录
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

                level = root.replace(project_root, '').count(os.sep)
                indent = ' ' * 4 * (level)
                outfile.write(f'{indent}{os.path.basename(root)}/\n')
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    if os.path.splitext(f)[1] in TARGET_EXTENSIONS and f not in IGNORE_FILES:
                        outfile.write(f'{subindent}{f}\n')

            outfile.write("\n\n" + "=" * 50 + "\n\n")
            outfile.write("File Contents:\n")
            outfile.write("==================\n\n")

            # 第二遍遍历：提取内容
            for root, dirs, files in os.walk(project_root):
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

                for file in files:
                    file_ext = os.path.splitext(file)[1]

                    if file_ext in TARGET_EXTENSIONS and file not in IGNORE_FILES:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, project_root)

                        try:
                            with open(file_path, 'r', encoding='utf-8') as infile:
                                content = infile.read()

                                # 写入格式化的分隔符和文件名
                                outfile.write(f"START OF FILE: {rel_path}\n")
                                outfile.write("-" * 50 + "\n")
                                outfile.write(content)
                                outfile.write("\n" + "-" * 50 + "\n")
                                outfile.write(f"END OF FILE: {rel_path}\n\n")

                                print(f"已处理: {rel_path}")
                                total_files += 1
                        except Exception as e:
                            print(f"无法读取文件 {rel_path}: {e}")

        print(f"\n成功! 共处理 {total_files} 个文件。")
        print(f"结果已保存至: {OUTPUT_FILE}")

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    collect_code()