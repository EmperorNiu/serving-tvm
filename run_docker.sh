nohup sudo docker run -p 8502:8500 -p 8503:8501 --name tfserving_tvm -t tensorflow/serving:tvm &
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=/root/model_config.json
tensorflow_model_server --port=6003  --rest_api_port=6004 

