#1.ssh连接
    ssh-secure shell
    1.安装ssh service
        sudo apt-get install open-ssh
    2.启动ssh服务
        sudo service ssh start
    3.修改ssh配置
        sudo vi /etc/ssh/sshd_config
        permitrootlogin yes
        permitpasswdauthentication yes
        ...
    4.ssh连接
        ssh -p 22 root@ipaddress
        enter passwd

#2.scp
    secure copy
    远程拷贝文件
    scp 本地文件路径 远程机器的用户名@远程机器的ip地址:远程机器的文件存储地址
