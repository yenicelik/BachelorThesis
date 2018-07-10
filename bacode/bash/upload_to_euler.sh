# rsync -r /Users/davidal/GoogleDrive/BachelorThesis/code yedavid@euler.ethz.ch:/cluster/home/yedavid/BachelorThesis/ --progress
rsync -r /Users/davidal/GoogleDrive/BachelorThesis/bacode/ yedavid@supermodularity.inf.ethz.ch:/home/yedavid/BachelorThesis/bacode/ --progress -av
rsync -r /Users/davidal/febo/ yedavid@supermodularity.inf.ethz.ch:/home/yedavid/febo/ --progress -v

# rsync -avL --progress -e "ssh -i /Users/davidal/.ssh/AWS_BA.pem" \
#        /Users/davidal/GoogleDrive/BachelorThesis/bacode/ \
#        ubuntu@ec2-52-73-132-192.compute-1.amazonaws.com:/home/yedavid/BachelorThesis/bacode/

# rsync -avL --progress -e "ssh -i /Users/davidal/.ssh/AWS_BA.pem" \
#        /Users/davidal/febo/ \
#        ubuntu@ec2-52-73-132-192.compute-1.amazonaws.com:/home/yedavid/febo/