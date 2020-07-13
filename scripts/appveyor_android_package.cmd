git clone https://github.com/LeelaChessZero/chessenginesupport-androidlib.git --branch lc0 --single-branch oex
cd oex
git checkout 0c02a8893b9c57ec57b40569ba60625912c6d32f
cd ..
perl -e "printf '%%sLc0!', pack('V', -s 'c:/cache/%NET%.pb.gz')" >tail.bin
copy /y /b arm64-v8a\lc0+c:\cache\%NET%.pb.gz+tail.bin oex\LeelaChessEngine\leelaChessEngine\src\main\jniLibs\arm64-v8a\liblc0.so
copy /y /b armeabi-v7a\lc0+c:\cache\%NET%.pb.gz+tail.bin oex\LeelaChessEngine\leelaChessEngine\src\main\jniLibs\armeabi-v7a\liblc0.so
set ANDROID_HOME=C:\android-sdk-windows
appveyor DownloadFile https://dl.google.com/android/repository/sdk-tools-windows-3859397.zip
7z x sdk-tools-windows-3859397.zip -oC:\android-sdk-windows > nul
yes | C:\android-sdk-windows\tools\bin\sdkmanager.bat --licenses
cd oex\LeelaChessEngine
sed -i "s/591226/%NET%/" leelaChessEngine/src/main/res/values/strings.xml
sed -i "/versionCode/ s/1/%APPVEYOR_BUILD_NUMBER%/" leelaChessEngine/src/main/AndroidManifest.xml
sed -i "s/0.25dev/%APPVEYOR_REPO_TAG_NAME%/" leelaChessEngine/src/main/AndroidManifest.xml
call gradlew.bat assemble
copy leelaChessEngine\build\outputs\apk\debug\leelaChessEngine-debug.apk ..\..\lc0-%APPVEYOR_REPO_TAG_NAME%-android.apk
