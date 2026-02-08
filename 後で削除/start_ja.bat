@echo off
chcp 932 > nul
echo ========================================
echo  ﾏﾙﾁﾓｰﾀﾞﾙRAGｼｽﾃﾑ 起動
echo ========================================
echo.

REM 仮想環境の存在チェック
if not exist venv (
    echo [ｴﾗｰ] 仮想環境が見つかりません
    echo まず setup.bat または setup_ja.bat を実行してください
    echo.
    pause
    exit /b 1
)

echo [処理] 仮想環境をｱｸﾃｨﾍﾞｰﾄ中...
call venv\Scripts\activate.bat

echo [処理] 環境ﾁｪｯｸ中...
python run.py

REM エラーが発生した場合
if errorlevel 1 (
    echo.
    echo [情報] ｴﾗｰが発生しました
    echo ﾄﾗﾌﾞﾙｼｭｰﾃｨﾝｸﾞ:
    echo 1. setup.bat を再度実行してみてください
    echo 2. .env ﾌｧｲﾙにAPIｷｰが設定されているか確認してください
    echo 3. または、ｱﾌﾟﾘ起動後にｻｲﾄﾞﾊﾞｰからAPIｷｰを入力してください
    echo.
    pause
)
