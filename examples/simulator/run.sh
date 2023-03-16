
PYTHONPATH=.:$PYTHONPATH python examples/cases_tetris.py \
    --premise interlace \
    --ndevs 4 --nmicros 4 --memory 10 --save figures/ \
    > figures/interlace-4dev-4nmb-10mem.log

PYTHONPATH=.:$PYTHONPATH python examples/cases_tetris.py \
    --premise vshape \
    --ndevs 4 --nmicros 4 --memory 4 --save figures/ \
    > figures/vshape-4dev-4nmb-4mem.log

PYTHONPATH=.:$PYTHONPATH python examples/cases_tetris.py \
    --premise finetune \
    --ndevs 4 --nmicros 4 --memory 4 --save figures/ \
    > figures/finetune-4dev-4nmb-4mem.log