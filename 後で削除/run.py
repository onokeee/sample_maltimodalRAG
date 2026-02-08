"""
簡易起動スクリプト
このスクリプトを実行すると、必要な環境チェックを行ってからStreamlitアプリを起動します
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """環境チェック"""
    print("=" * 50)
    print("環境チェック開始")
    print("=" * 50)
    
    # Pythonバージョンチェック
    python_version = sys.version_info
    print(f"✓ Pythonバージョン: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("✗ Python 3.8以上が必要です")
        return False
    
    # 必要なパッケージチェック
    required_packages = [
        'streamlit',
        'llama_index',
        'chromadb',
        'openai'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} インストール済み")
        except ImportError:
            print(f"✗ {package} が見つかりません")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n必要なパッケージがインストールされていません。")
        print("以下のコマンドを実行してください:")
        print(f"  pip install -r requirements.txt")
        return False
    
    # .envファイルチェック
    env_file = Path('.env')
    if not env_file.exists():
        print("\n✗ .envファイルが見つかりません")
        print("  .env.exampleを.envにコピーして、APIキーを設定してください")
        print("  または、アプリ起動後にサイドバーから入力してください")
    else:
        print("✓ .envファイル確認")
    
    # ディレクトリチェック
    required_dirs = ['uploaded_data', 'chroma_db']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"✓ {dir_name}/ ディレクトリを作成しました")
        else:
            print(f"✓ {dir_name}/ ディレクトリ確認")
    
    print("\n" + "=" * 50)
    print("環境チェック完了！")
    print("=" * 50)
    return True

def main():
    """メイン処理"""
    print("\n🔍 マルチモーダルRAGシステム 起動スクリプト\n")
    
    # 環境チェック
    if not check_environment():
        print("\n環境チェックに失敗しました。上記のエラーを解決してください。")
        sys.exit(1)
    
    print("\nStreamlitアプリを起動します...\n")
    
    # Streamlitアプリの起動
    try:
        subprocess.run(['streamlit', 'run', 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\nアプリケーションを終了します。")
    except subprocess.CalledProcessError as e:
        print(f"\nエラーが発生しました: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nstreamlitコマンドが見つかりません。")
        print("以下のコマンドでインストールしてください:")
        print("  pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
