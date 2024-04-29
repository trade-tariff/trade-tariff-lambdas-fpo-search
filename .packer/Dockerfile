FROM hashicorp/packer
COPY . $HOME/app
WORKDIR $HOME/app
RUN packer init template.pkr.hcl
ENTRYPOINT ["packer"]
