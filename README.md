# Brain-Abscesses

How to create brain mask using hd-bet

pip install hd-bet

hd-bet --input ./dataset/registered/240/FLAIR.nii.gz --output ./dataset/masked/240/FLAIR_brain.nii.gz --save_bet_mask
OR
hd-bet --input ./dataset/brain_mask_input --output ./dataset/brain_masks --save_bet_mask  --no_bet_image
