FROM golang:1.22 AS build
ADD --checksum=sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef https://example.com/tools/rustup-init.sh /tmp/rustup-init.sh
RUN sh /tmp/rustup-init.sh -y
WORKDIR /src
COPY . .
RUN go build -o /out/builder ./build

FROM gcr.io/distroless/base-debian12@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
COPY --from=build /out/builder /builder
USER nonroot
ENTRYPOINT ["/builder"]
