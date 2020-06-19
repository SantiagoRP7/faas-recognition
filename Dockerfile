FROM FROM python:3-slim

ADD https://github.com/alexellis/faas/releases/download/0.5.5-alpha/fwatchdog /usr/bin


RUN chmod +x /usr/bin/fwatchdog


COPY index.py index.py

COPY pyfunction pyfunction

RUN touch ./pyfunction/__init__.py

WORKDIR /root/pyfunction/
COPY pyfunction/requirements.txt	.
RUN pip install -r requirements.txt

WORKDIR /root/

ENV fprocess="python index.py"

HEALTHCHECK --interval=1s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
