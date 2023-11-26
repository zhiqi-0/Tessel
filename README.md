# Tessel

Tessel is a schedule composer that searches efficient pipeline schedules for any operator placement strategies of large DNN models.

## Install

```bash
pip install -e .
```

For runtime part, this repo is built on MagicCube of branch `zhiqi/main`.

## Examples

`examples/simulator/cases_tessel` provides various examples of operator placement strategies. You can try out with an example like:

```bash
python examples/simulator/cases_tessel.py \
    --premise mshape --ndevs 4 \
    --nmicros 6 --memory 6 \
    --save mshape.4stages.sched.json
```


## Contributions

All contributions are welcomed! Please issue PR for new cases or new features. 

## License

This project is under MIT license.