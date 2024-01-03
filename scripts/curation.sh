
data=${1}  # support {'c3', 'c12', 'y14'}
model=${2}  # support {'clip-vit-b-16', 'clip-vit-b-32', 'clip-vit-l-14', 'openclip-vit-h-14'} 

# <1> extraction
python curation_utils/concept_extract.py --root /home/xingyun/GroupViT --data ${data} --model ${model}

# <2> forward
python curation_utils/concept_forward.py --root /home/xingyun/GroupViT --model ${model} 

# <3> curation
python curation_utils/curation.py --root /home/xingyun/GroupViT --data ${data} --model ${model} --k 16
