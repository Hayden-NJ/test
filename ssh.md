SSH为建立在应用层和传输层基础上的安全协议。这是一种网络协议，用于计算机之间的加密登录。

需要指出的是，SSH只是一种协议，存在多种实现，既有商业实现，也有开源实现。本文针对的实现是OPENSSH，此外，本文只讨论SSH在Linux Shell中的用法。如果要在Windows系统中使用SSH，会用到另一种软件[PuTTY](http://www.chiark.greenend.org.uk/~sgtatham/putty)，这需要另文介绍。

登录远程主机：

ssh user@host

默认登录端口是22，可-p修改参数： 

ssh -p 2222 user@host

如果你是第一次登录对方主机，系统会出现下面的提示： 

$ ssh user@host

The authenticity of host 'host (12.18.429.21)' can't be established.

RSA key fingerprint is 98:2e:d7:e0:de:9f:ac:67:28:c2:42:2d:37:16:58:4d.

Are you sure you want to continue connecting (yes/no)? yes

Warning: Permanently added 'host,12.18.429.21' (RSA) to the list of known hosts.

生成秘钥：运行结束以后，在$HOME/.ssh/目录下，会新生成公钥和私钥：id_rsa.pub和id_rsa。

ssh-keygen

将公钥传送到远程主机host上面：

ssh-copy-id user@host

git bash生成秘钥：会询问文件存放地址和是否需要密码。

ssh-keygen -t rsa -C '281831290@qq.com'

指定文件夹和文件名生成密钥

ssh-keygen -t rsa -C "你的注册邮箱” -f ~/.ssh/id_rsa_one

验证密钥是否正常工作：第一次验证会提示The authenticity of host 'github.com (192.30.255.112)' can't be established.等后续。

ssh -T git@gitubcom

ssh -vT git@gitubcom





### 多个sshkey对应多个不同的github账号

多个github账号之间想共享公钥？这个就是github限制了。ssh key是用来辨别账号的，一般一个人只需要一个账号而已，这个账号若想直接参与另外一个账号或组织的项目， 让那个项目把你加为成员即可。你还是用这一个账号。一般不需要多个账号。

##### 将新生成的密钥添加到SSH agent中（因为系统默认只读取id_rsa，为了让SSH识别新的私钥，需将其添加到SSH agent中）

ssh-agent bash

ssh-add ~/.ssh/abc

```
# 可以通过 ssh-add -l 来确私钥列表
$ ssh-add -l
$ ssh-add -L
# 可以通过 ssh-add -D 来清空私钥列表
$ ssh-add -D
```

##### 在~/.ssh/目录下进行config文件的配置(如果没有就新建一个,不用后缀名)

```
主要是HostName和IdentityFile要改,HostName是服务器域名，IdentityFile 就是密钥的地址了
# 这个是原来的key
# github
Host github.com
HostName github.com
PreferredAuthentications publickey
IdentityFile ~/.ssh/id-rsa
	
# 这个是新加的key
# github_2
Host github_1.com   (此处的host名是自己取的,你也可以自己改)
HostName github.com		(gitlab的话写gitlab.com? //这里填你们公司的git网址即可)
PreferredAuthentications publickey		
IdentityFile ~/.ssh/abc		(这是你的key的路径名)
```

##### 在你需要连接的github的settings里配置sshkey

将新生成的公钥(.pub后缀)复制过去

##### 测试

```
ssh -T git@github.com
```

Hi stefzhlg! You've successfully authenticated, but GitHub does not provide shell access.

就表示成功的连上github了.也可以试试链接公司的gitlab.

##### 修改克隆或者关联远程仓库的地址(关键)

git remote add origin git@github.com:name/project.gi

##### 然后在不同的仓库下设置局部的用户名和邮箱?另外一个人写的，需验证。

 比如在公司的repository下`git config user.name "yourname" git config user.email "youremail"` 在自己的github的仓库在执行刚刚的命令一遍即可。

配置git仓库用户名和邮箱

##### 可以为每个git仓库配置单独的用户名和邮箱，没有配置的则使用全局的用户名和邮箱？另一个人写的，需验证。s

```bash
$ git config –global user.name “global_username”
$ git config –global user.email “global_email.com”
```













### github Key is already in use

出现这个的原因是你在以前也用过GitHub, 并且提交了你的密钥. 这个时候你可以通过在命令行里输入:

```
ssh -T -i ~/.ssh/id_rsa git@github.com
```

来查看到底是哪一个账户在使用此密钥，会出现如下提示:

```
Hi <XXX>! You've successfully authenticated, but GitHub does not provide shell access.
```

就是这个XXX账号， 占用了当前sshkey, 登陆这个账号，删除掉sshkey就行了

##### 当然，官方也说了找到在哪个地方使用了该键：

```bash
ssh -T -ai ~/.ssh/id_rsa git@github.com
# Connect to GitHub using a specific ssh key
Hi username! You've successfully authenticated, but GitHub does notprovide shell access.
```



### ~/.ssh/known_hosts文件的作用

ssh会把你每个你访问过计算机的公钥(public key)都记录在~/.ssh/known_hosts。当下次访问相同计算机时，OpenSSH会核对公钥。如果公钥不同，OpenSSH会发出警告， 避免你受到DNS Hijack之类的攻击。我在上面列出的情况，就是这种情况。 

原因：一台主机上有多个Linux系统，会经常切换，那么这些系统使用同一ip，登录过一次后就会把ssh信息记录在本地的~/.ssh/known_hsots文件中，切换该系统后再用ssh访问这台主机就会出现冲突警告，需要手动删除修改known_hsots里面的内容。 

有以下两个解决方案： 
\1. 手动删除修改known_hsots里面的内容； 
\2. 修改配置文件“~/.ssh/config”，加上这两行，重启服务器。 
  StrictHostKeyChecking no 
  UserKnownHostsFile /dev/null 

优缺点： 
\1. 需要每次手动删除文件内容，一些自动化脚本的无法运行（在SSH登陆时失败），但是安全性高； 
\2. SSH登陆时会忽略known_hsots的访问，但是安全性低；





##### 其他参考

参考阮一峰

http://www.ruanyifeng.com/blog/2011/12/ssh_remote_login.html















