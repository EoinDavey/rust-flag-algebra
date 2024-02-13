.PHONY: clean default

default:
	@echo "empty"

clean:
	rm -r ./certificate ./data/ ./target/ ./*.sdpa ./report.html
