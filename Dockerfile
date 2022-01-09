# use debian-based python image to match travis CI matplotlib outputs
FROM python:3.8

RUN mkdir /src
WORKDIR /src

COPY requirements_dev.txt /src/
RUN pip install -r requirements_dev.txt

# COPY . /src/
# no, this will be mounted in, because all local scripts aim to modify the working directory

CMD ["/bin/bash"]
