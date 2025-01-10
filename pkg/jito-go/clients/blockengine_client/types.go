package blockengine_client

import (
	"github.com/newruscult/xo-sniper/pkg/jito-go/pkg"
	"github.com/newruscult/xo-sniper/pkg/jito-go/proto"
	"google.golang.org/grpc"
)

type Relayer struct {
	GrpcConn *grpc.ClientConn

	Client proto.BlockEngineRelayerClient

	Auth *pkg.AuthenticationService

	ErrChan <-chan error // ErrChan is used for dispatching errors from functions executed within goroutines.
}

type Validator struct {
	GrpcConn *grpc.ClientConn

	Client proto.BlockEngineValidatorClient

	Auth *pkg.AuthenticationService

	ErrChan <-chan error // ErrChan is used for dispatching errors from functions executed within goroutines.
}
