# gitbook环境配置
推荐使用nvm (https://github.com/nvm-sh/nvm)

nvm install 10.14.1 # 安装10版本的nodejs，高版本会出错
nvm use 10.14.1
npm install -g gitbook-cli


# 修改内容
main branch 只是用来在github pages上部署，
要修改内容需要在dev branch。

修改完成后
#编译gitbook
gitbook build 
#编译并本地运行gitbook
gitbook serve

编译之后出现一个_book目录，之后可以用下列命令进行上传，更新github pages。注意路径和branch，由于用了push -f，有可能导致数据丢失。

cd _book
git init
git branch -M main
git remote add origin git@github.com:circuitnet/circuitnet.github.io.git
git add .
git commit -m 'update'
git push -uf origin main


主目录下的SUMMARY.md是gitbook的目录，README.md是gitbook的首页，其他页分散在各个目录里，可以在SUMMARY.md中查看。