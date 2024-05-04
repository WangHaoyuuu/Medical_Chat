# Medical_Chat
# 如何运行
1. 创建一个.env文件，在里面填充各个平台的apikey
2. conda create -n MChat python==3.9 -y
3. conda activate MChat
4. pip install -r requirements.txt
5. pip install httpx[socks]
6. cd WHYserve 
7. python test_run_gradio.py