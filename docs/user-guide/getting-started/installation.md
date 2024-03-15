# Installation

*Note: this section applies in the current form only to Kaiko-internal user testing and will be revised for the public package when publishing eva*


- Create and activate a virtual environment with Python 3.10+

- Install *eva* and the *eva-vision* package with:

```
pip install --index-url https://nexus.infra.prd.kaiko.ai/repository/python-all/simple 'kaiko-eva[vision]'
```

- To be able to use the existing configs, download them into directory where you installed *eva*. You can get them from our blob storage with:

```
azcopy copy https://kaiko.blob.core.windows.net/long-term-experimental/eva/configs . --recursive=true
```

(Alternatively you can also download them from the [*eva* GitHub repo](https://github.com/kaiko-ai/eva/tree/main))


## Run *eva*

Now you are all setup and you can start running *eva* with:
```
python -m eva <subcommand> --config <path-to-config-file>
```
To learn how the subcommands and configs work, we recommend you familiarize yourself with [How to use *eva*](how_to_use.md) and then proceed to running *eva* with the [Tutorials](../tutorials/offline_vs_online.md).
