IMAGE_NAME=rydercalmdown/dog-detector
IMAGE_VERSION=latest
RASPBERRY_PI_IP=10.0.0.86
RASPBERRY_PI_USERNAME=pi


.PHONY: install
install:
	@echo "Installing locally with docker"
	@docker build -t $(IMAGE_NAME):$(IMAGE_VERSION) .

.PHONY: build
build:
	@make install

.PHONY: run
run:
	@docker run -t -v $(shell pwd)/src/:/code/ $(IMAGE_NAME):$(IMAGE_VERSION)

.PHONY: install-local
install-local:
	@virtualenv -p python3 env && . env/bin/activate && pip install -r src/requirements.txt

.PHONY: run-local
run-local:
	@source env/bin/activate && cd src && python app.py

.PHONY: install-on-pi
install-on-pi:
	@echo "Installing raspberry pi"

.PHONY: copy
copy:
	@echo "Copying to raspberry pi"
	rsync -r $(shell pwd) --exclude env --exclude data $(RASPBERRY_PI_USERNAME)@$(RASPBERRY_PI_IP):/home/$(RASPBERRY_PI_USERNAME)

.PHONY: focus
focus:
	@echo "Taking camera photo for focus"
	@ssh $(RASPBERRY_PI_USERNAME)@$(RASPBERRY_PI_IP) raspistill -o /home/pi/dog_detector/focus.jpg
	@scp $(RASPBERRY_PI_USERNAME)@$(RASPBERRY_PI_IP):/home/pi/dog_detector/focus.jpg ./focus.jpg

.PHONY: console
console:
	@echo "Connecting to raspberry pi"
	@ssh $(RASPBERRY_PI_USERNAME)@$(RASPBERRY_PI_IP)
