# Keeping AMIs updated

We build custom AMIs based off of AWS Deep Learning AMIS which are continuously deprecated and upgraded by AWS.

We upgrade these AMIs to do two things:

1. Raise the ceiling on supported versions of torch during training
2. Unblock security patches in the underlying libraries and languages we use

_Supported_ here essentially means:

1. Enables compatible use of the user land CUDA driver APIs (without these we'd rely on CPU, only, which would slow training down to 8 hours compared to 1 hour with GPU drivers)
2. Unblocks newer versions of python, C and C++ for native binaries (all of which we use heavily)

As well as compatibility, it is important that we keep them up-to-date on a regular basis to avoid the risks associated with older libraries and to get the benefits of any upgrades to these libraries.

These AMIs are pivotal to the smooth running of the FPO training process.

## Summary of updating an AMI

The process of updating AMIs summarised is:

1. Find the latest AMI from the AWS AMI catalogue
2. Verify compatible instance types for this AMI
3. Pull the latest Amazon Linux and verify versions of the latest languages and drivers locally
4. Upgrade all packages in requirements.txt
5. Provision the full environment locally
6. Provision a new AMI in GithubActions via EC2 and packer

Prerequisites

1. AWS access and credentials loaded in a shell in your local machine
2. Docker
3. Properly configured python
4. Virtual env setup `python -m venv venv && source venv/bin/activate`

> _Sanity check local provisioning before making changes_
> `make test-provision` will run a full provision of an Amazon Linux container using the same script
> that gets executed by packer in EC2 when building our AMI
>
> If you're experiencing issues, here, and need to debug you can run `make shell-provision` and run `./provision` in an shell-interactive environment.

## Finding the latest available AMI

```sh
# Fetch a list of the latest AMIs ordered by CreationDate
.github/bin/getlatestamis
# Searching for latest Deep Learning Base AMI (AL2023) in us-east-1...
# -------------------------------------------------------------------------------------------------------------------------------
# |                                                       DescribeImages                                                        |
# +-----------------------+------------------------------------------------------------------------+----------------------------+
# |  ami-01a45bc7e19f2bb26|  Deep Learning Base AMI with Single CUDA (Amazon Linux 2023) 20251219  |  2025-12-19T16:34:04.000Z  |
# |  ami-09f5f5e0999e6246c|  Deep Learning Base AMI with Single CUDA (Amazon Linux 2023) 20251216  |  2025-12-16T17:53:30.000Z  |
# |  ami-081e81b655efbb639|  Deep Learning Base AMI with Single CUDA (Amazon Linux 2023) 20251212  |  2025-12-12T15:10:10.000Z  |
# |  ami-035b44de1129006c3|  Deep Learning Base AMI with Single CUDA (Amazon Linux 2023) 20251209  |  2025-12-09T15:04:21.000Z  |
# |  ami-00a3a6192ba06e9ae|  Deep Learning Base AMI with Single CUDA (Amazon Linux 2023) 20251205  |  2025-12-05T15:18:09.000Z  |
# +-----------------------+------------------------------------------------------------------------+----------------------------+
```

## Finding the latest compatible versions of the compiled languages we use

Run `make shell-provision` to pull and get an interactive shell in the environment we're going to be running in our AMI.

```sh
yum search gcc
# gcc.x86_64 : Various compilers (C, C++, Objective-C, ...)
# gcc-c++.x86_64 : C++ support for GCC
# gcc14.x86_64 : Various compilers (C, C++, Objective-C, ...)
# gcc14-c++.x86_64 : C++ support for GCC
# ...
```

Historically the gcc that's installed when you run `yum install gcc` is always older than the version in the name matched section of the `yum search` output. This is the one we want.

```sh
yum install -y gcc14 gcc14-c++
```

Update the `.packer/provision` script to install and link the new gcc versions. This is usually just a version number substitution in the provision script whilst the rest of the name stays the same.

## Updating python

We use pyenv to configure python in the `provision` script. You'll want to update the version in this script (`.packer/provision`) and validate your changes.

## Updating python dependencies

We have two places to update python dependencies.

This is due to the fact that dependencies are different whether you are training a new model or classifying goods using an existing model.

- `requirements.txt` - used in the provision script to install the dependencies with torch binaries for CUDA
- `requirements_lambda.txt` - used in the deployed lambda behind API gateway and installed in the Docker build process

Typically we install the latest versions of everything and work backwards if conflicts are found in the dependency graph resolver.

## Validating the change

As we've done previously, we'll now want to integrate our version change against a fully provisioned and as close to the real environment as possible docker container.

Assuming the `.packer/provision` script is up-to-date with your changes you can run:

```sh
make test-provision
```

The result of this should be a successful installation of the all of the dependencies needed for doing a training run.

## Updating the AMI

With the latest version of the Deep Learning AMI we found earlier, we need to replace the AMI ID in the `.packer/template.pkr.hcl` with this updated version.

To trigger a build in the GithubActions development workflow we'll need to make sure to bump the `.packer/ami-version.txt` file which follows semantic versioning.

> Typically I bump the major version when we're on completely different generations of AMI or completely new instance types and only minor version updates for `requirements.txt` python or C/C++ updates.

Open a pull request and this will trigger a full build and deployment of the new AMI which is automatically consumable by our model training process.

> It's worth also bumping the version in `./search-config.toml` just to prove a full integration since occasionally AMI upgrades alone work but change something unexpected downstream that needs review.

You'll want to consider updating the index URL in the provision script to point to the latest CUDA toolkit version if applicable.

You can determine the compatible CUDA toolkit version from the AWS Deep Learning AMI documentation.

https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html

## Conclusion

You've now successfully updated the AMI we use for training models in FPO!

It is very much an iterative process of trying versions/dependencies until everything works together so don't be disheartened if it takes a few attempts to get everything working.
