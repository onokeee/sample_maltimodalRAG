@echo off
chcp 932 > nul
echo ========================================
echo  ﾏﾙﾁﾓｰﾀﾞﾙRAGｼｽﾃﾑ ｾｯﾄｱｯﾌﾟ
echo ========================================
echo.

REM 仮想環境の存在チェック
if exist venv (
    echo [情報] 既存の仮想環境が見つかりました
    choice /C YN /M "既存の仮想環境を削除して再作成しますか"
    if errorlevel 2 goto activate_existing
    if errorlevel 1 goto delete_venv
) else (
    goto create_venv
)

:delete_venv
echo.
echo [処理] 既存の仮想環境を削除中...
rmdir /s /q venv
echo [完了] 既存の仮想環境を削除しました
goto create_venv

:create_venv
echo.
echo [処理] Python仮想環境を作成中...
python -m venv venv
if errorlevel 1 (
    echo [ｴﾗｰ] 仮想環境の作成に失敗しました
    echo Pythonが正しくｲﾝｽﾄｰﾙされているか確認してください
    pause
    exit /b 1
)
echo [完了] 仮想環境を作成しました
goto activate_and_install

:activate_existing
echo [処理] 既存の仮想環境を使用します
goto activate_and_install

:activate_and_install
echo.
echo [処理] 仮想環境をｱｸﾃｨﾍﾞｰﾄ中...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ｴﾗｰ] 仮想環境のｱｸﾃｨﾍﾞｰﾄに失敗しました
    pause
    exit /b 1
)

echo [完了] 仮想環境をｱｸﾃｨﾍﾞｰﾄしました
echo.
echo [処理] pipをｱｯﾌﾟｸﾞﾚｰﾄﾞ中...
python -m pip install --upgrade pip
echo.
echo [処理] 必要なﾊﾟｯｹｰｼﾞをｲﾝｽﾄｰﾙ中...
echo この処理には数分かかる場合があります...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ｴﾗｰ] ﾊﾟｯｹｰｼﾞのｲﾝｽﾄｰﾙに失敗しました
    pause
    exit /b 1
)

echo.
echo ========================================
echo  ｾｯﾄｱｯﾌﾟ完了!
echo ========================================
echo.
echo 次のｽﾃｯﾌﾟ:
echo 1. .env.example を .env にｺﾋﾟｰして、APIｷｰを設定
echo 2. start_ja.bat を実行してｱﾌﾟﾘを起動
echo.
echo または、以下のｺﾏﾝﾄﾞで起動:
echo   venv\Scripts\activate
echo   streamlit run app.py
echo.
pause
