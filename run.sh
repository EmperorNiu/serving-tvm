# docker cp
cd /home/yaniu/tfserving/serving-docker-2.6/tensorflow_serving/servables/tvm
sudo docker cp ./tensorflow_serving/servables/tvm cba:/tensorflow-serving/tensorflow_serving/servables
sudo docker cp ./third_party/tvm cba:/tensorflow-serving/third_party
sudo docker cp ./tensorflow_serving/workspace.bzl cba:/tensorflow-serving/tensorflow_serving
# docker build
sudo docker exec -it cba bash
bazel build --config=release --experimental_cc_shared_library tensorflow_serving/...

# proxy
./clash &
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890