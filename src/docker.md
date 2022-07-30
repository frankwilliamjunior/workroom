#1. docker安装
    1.使用官方脚本自动安装
    curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
    2.国内daocloud一键安装
    curl -sSL https://get.daocloud.io/docker | sh
    3.手动安装
        卸载旧版本
        sudo apt-get remove docker docker-engine docker.io containerd runc
        更新apt包索引
        sudo apt-get update
        安装apt依赖包
        sudo apt-get install ca-certificates curl gnupg lsb-release
        添加官方GPG密钥
        sudo mkdir -p /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        设置仓库
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \ https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        安装docker engine
        sudo apt-get update
        sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
    4. 安装指定docker版本
        查询可用列表
        apt-cache madison docker-ce
        安装指定版本
        sudo apt-get install docker-ce=<VERSION_STRING> docker-ce-cli=<VERSION_STRING> \
        containerd.id docker-compose-plugin
#2.运行docker
    sudo service docker start

#3.拉取镜像
    docker登陆
    docker login
    docker pull image-name(公开docker hub中的镜像名)

#4.启动容器(镜像)，并以命令行进入
    docker run -it image-name /bin/bash

#5.查看所有的容器
    docker ps -a
    启动已停止的容器
    docker start <容器id>

#6.进入容器
    进入容器,退出则停止
    docker attach <image-name>
    进入容器,退出并不停止
    docker exec -it <image-name> /bin/bash

#7.容器删除、镜像删除
    docker rm <容器id>
    docker rmi <镜像id>

#8.端口映射
    将内部容器的端口随机映射到主机的端口
        docker run -P <image-name> <other-command>
    将内部容器的端口映射到指定的主机端口  宿主机ip可以省略
    docker run -itd -p (宿主机ip):宿主机端口:容器端口 --name 容器名 镜像名 /bin/bash

#9.宿主机的文件夹挂载映射
    docker run -it -v <宿主机文件夹>:<容器文件夹> <image-name>
    