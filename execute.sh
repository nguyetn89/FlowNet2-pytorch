dataset=$1
subset=$2
if [ "$dataset" == "UCSDped2" ]; then
    recalc=1
    echo "Processing dataset ${dataset}..."
    if [[ "$subset" == "train" || "$subset" == "both" ]]; then
        echo "Processing training data..."
        in_path="/home/nguyetn/projects/def-jeandiro/nguyetn/Anomaly/datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
        for i in $(seq -w 001 016)
        do
            echo "clip ${i}"
            python3 run_flow_video.py --in_path="${in_path}/Train${i}" --out_file="${in_path}/Train${i}_full.npy" --scale 3 --recalc ${recalc}
        done
    fi
    if [[ "$subset" == "test" || "$subset" == "both" ]]; then
        echo "Processing test data..."
        in_path="/home/nguyetn/projects/def-jeandiro/nguyetn/Anomaly/datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"
        for i in $(seq -w 001 012)
        do
            echo "clip ${i}"
            python3 run_flow_video.py --in_path="${in_path}/Test${i}" --out_file="${in_path}/Test${i}_full.npy" --scale 3 --recalc ${recalc}
        done
    fi
elif [ "$dataset" == "Avenue" ]; then
    recalc=1
    echo "Processing dataset ${dataset}..."
    if [[ "$subset" == "train" || "$subset" == "both" ]]; then
        echo "Processing training data..."
        in_path="/home/nguyetn/projects/def-jeandiro/nguyetn/Anomaly/datasets/Avenue_Dataset/training_videos"
        for i in $(seq -w 01 16)
        do
            echo "clip ${i}"
            python3 run_flow_video.py --in_path="${in_path}/${i}.avi" --out_file="${in_path}/${i}_full.npy" --scale 3 --recalc ${recalc}
        done
    fi
    if [[ "$subset" == "test" || "$subset" == "both" ]]; then
        echo "Processing test data..."
        in_path="/home/nguyetn/projects/def-jeandiro/nguyetn/Anomaly/datasets/Avenue_Dataset/testing_videos"
        for i in $(seq -w 01 21)
        do
            echo "clip ${i}"
            python3 run_flow_video.py --in_path="${in_path}/${i}.avi" --out_file="${in_path}/${i}_full.npy" --scale 3 --recalc ${recalc}
        done
    fi
elif [ "$dataset" == "ShanghaiTech" ]; then
    recalc=0
    echo "Processing dataset ${dataset}..."
    if [[ "$subset" == "train" || "$subset" == "both" ]]; then
        echo "Processing training data..."
        in_path="/home/nguyetn/projects/def-jeandiro/nguyetn/Anomaly/datasets/shanghaitech/training/videos"
        for file in ${in_path}/*.avi
        do
            echo "clip ${file}"
            python3 run_flow_video.py --in_path="${file%.*}.avi" --out_file="${file%.*}_full.npy" --scale 3 --recalc ${recalc}
        done
    fi
    if [[ "$subset" == "test" || "$subset" == "both" ]]; then
        echo "Processing test data..."
        in_path="/home/nguyetn/projects/def-jeandiro/nguyetn/Anomaly/datasets/shanghaitech/testing/frames"
        for file in ${in_path}/*/
        do
            echo "clip ${file}"
            python3 run_flow_video.py --in_path="${file%/}" --out_file="${file%/}_full.npy" --scale 3 --recalc ${recalc}
        done
    fi
else
    echo "Unknown dataset ${dataset}"
    exit 1
fi
