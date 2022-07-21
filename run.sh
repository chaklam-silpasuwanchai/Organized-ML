cd src

#set seed
declare -i seed=42

#run experiments of different configs
# python main.py -c fc1.yaml -s $seed   #run linear -> relu -> linear
python main.py -c cnn1.yaml -s $seed   #run cnn -> cnn -> linear