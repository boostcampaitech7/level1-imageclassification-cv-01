FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Update and install necessary packages
RUN apt-get update -y && apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libglib2.0-0 wget git curl tmux sudo python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python

# Set timezone to Asia/Seoul
RUN DEBIAN_FRONTEND=noninteractive ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    DEBIAN_FRONTEND=noninteractive dpkg-reconfigure -f noninteractive tzdata

# ssh 
# Clone repository and download data
# RUN git clone https://github.com/boostcampaitech7/level1-imageclassification-cv-01.git && \
#     cd level1-imageclassification-cv-01 && \
#     wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000307/data/data.tar.gz && \
#     tar -zxvf data.tar.gz && rm data.tar.gz

# Install Python requirements
# WORKDIR /level1-imageclassification-cv-01
# RUN pip install -r requirements.txt

# Set the default command
CMD ["/bin/bash"]