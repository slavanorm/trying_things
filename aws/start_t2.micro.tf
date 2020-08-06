provider "aws" {
	region		= "us-west-2"
	shared_credentials_file = "/home/v0/.aws/credentials"
}

# finding random ami
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/finding-an-ami.html
resource "aws_instance" "example" {
  ami           = "ami-0e34e7b9ca0ace12d"
  instance_type = "t2.micro"
}
