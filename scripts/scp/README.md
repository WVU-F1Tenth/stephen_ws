ssh server must be running for scp

Install openssh
```bash
sudo apt update
sudo apt install openssh-server
```

Check status
```bash
sudo systemctl status ssh
```

Start
```bash
sudo systemctl start ssh
```

Enable at boot
```bash
sudo systemctl enable ssh
```

Check port
ssh default is port 22
```bash
ss -tlnp | grep ssh
```