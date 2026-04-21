.PHONY: deploy-funasr deploy-whisper

deploy-funasr:
	STT_MODEL=paraformer-zh sh ./deploy.sh

deploy-whisper:
	STT_MODEL=whisper-large-v3-turbo sh ./deploy.sh
