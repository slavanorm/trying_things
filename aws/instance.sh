ID=$(aws ec2 describe-instances \
| jq -r '.Reservations[0].Instances[0].InstanceId')

if   [ "$1" = "start" ]; then
	aws ec2 start-instances --instance-id $ID
	echo running
elif [ "$1" = "create" ]; then
	terraform apply -auto-approve
elif [ "$1" = "stop" ]; then
	aws ec2 stop-instances --instance-id $ID
	echo stopping
else
	echo 'Usage: sh instance.sh start|create|stop'
	echo Requires: apt install jq
fi

