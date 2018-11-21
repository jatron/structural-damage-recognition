cd ~/Documents/

rm -rf ~/Documents/structural-damage-recognition

cp -r ~/personal/structural-damage-recognition ~/Documents/

cd structural-damage-recognition/tf_files/

rm -rf bottlenecks/ structure_photos/

cd ..

rm -rf my_virtual_env/ android/ ios/ .git/

cd ~/Documents/

zip -r "${1}.zip" structural-damage-recognition/

echo "created ${1}.zip"

cd ~/personal/structural-damage-recognition/
