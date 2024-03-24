import copy

import botocore.parsers
import botocore.serialize
from botocore.args import ClientArgsCreator

from .config import AioConfig
from .endpoint import AioEndpointCreator
from .regions import AioEndpointRulesetResolver
from .signers import AioRequestSigner


class AioClientArgsCreator(ClientArgsCreator):
    # NOTE: we override this so we can pull out the custom AioConfig params and
    #       use an AioEndpointCreator
    def get_client_args(
        self,
        service_model,
        region_name,
        is_secure,
        endpoint_url,
        verify,
        credentials,
        scoped_config,
        client_config,
        endpoint_bridge,
        auth_token=None,
        endpoints_ruleset_data=None,
        partition_data=None,
    ):
        final_args = self.compute_client_args(
            service_model,
            client_config,
            endpoint_bridge,
            region_name,
            endpoint_url,
            is_secure,
            scoped_config,
        )

        service_name = final_args['service_name']  # noqa
        parameter_validation = final_args['parameter_validation']
        endpoint_config = final_args['endpoint_config']
        protocol = final_args['protocol']
        config_kwargs = final_args['config_kwargs']
        s3_config = final_args['s3_config']
        partition = endpoint_config['metadata'].get('partition', None)
        socket_options = final_args['socket_options']

        signing_region = endpoint_config['signing_region']
        endpoint_region_name = endpoint_config['region_name']

        event_emitter = copy.copy(self._event_emitter)
        signer = AioRequestSigner(
            service_model.service_id,
            signing_region,
            endpoint_config['signing_name'],
            endpoint_config['signature_version'],
            credentials,
            event_emitter,
            auth_token,
        )

        config_kwargs['s3'] = s3_config

        # aiobotocore addition
        if isinstance(client_config, AioConfig):
            connector_args = client_config.connector_args
        else:
            connector_args = None

        new_config = AioConfig(connector_args, **config_kwargs)
        endpoint_creator = AioEndpointCreator(event_emitter)

        endpoint = endpoint_creator.create_endpoint(
            service_model,
            region_name=endpoint_region_name,
            endpoint_url=endpoint_config['endpoint_url'],
            verify=verify,
            response_parser_factory=self._response_parser_factory,
            max_pool_connections=new_config.max_pool_connections,
            proxies=new_config.proxies,
            timeout=(new_config.connect_timeout, new_config.read_timeout),
            socket_options=socket_options,
            client_cert=new_config.client_cert,
            proxies_config=new_config.proxies_config,
            connector_args=new_config.connector_args,
        )

        serializer = botocore.serialize.create_serializer(
            protocol, parameter_validation
        )
        response_parser = botocore.parsers.create_parser(protocol)

        ruleset_resolver = self._build_endpoint_resolver(
            endpoints_ruleset_data,
            partition_data,
            client_config,
            service_model,
            endpoint_region_name,
            region_name,
            endpoint_url,
            endpoint,
            is_secure,
            endpoint_bridge,
            event_emitter,
        )

        return {
            'serializer': serializer,
            'endpoint': endpoint,
            'response_parser': response_parser,
            'event_emitter': event_emitter,
            'request_signer': signer,
            'service_model': service_model,
            'loader': self._loader,
            'client_config': new_config,
            'partition': partition,
            'exceptions_factory': self._exceptions_factory,
            'endpoint_ruleset_resolver': ruleset_resolver,
        }

    def _build_endpoint_resolver(
        self,
        endpoints_ruleset_data,
        partition_data,
        client_config,
        service_model,
        endpoint_region_name,
        region_name,
        endpoint_url,
        endpoint,
        is_secure,
        endpoint_bridge,
        event_emitter,
    ):
        if endpoints_ruleset_data is None:
            return None

        # The legacy EndpointResolver is global to the session, but
        # EndpointRulesetResolver is service-specific. Builtins for
        # EndpointRulesetResolver must not be derived from the legacy
        # endpoint resolver's output, including final_args, s3_config,
        # etc.
        s3_config_raw = self.compute_s3_config(client_config) or {}
        service_name_raw = service_model.endpoint_prefix
        # Maintain complex logic for s3 and sts endpoints for backwards
        # compatibility.
        if service_name_raw in ['s3', 'sts'] or region_name is None:
            eprv2_region_name = endpoint_region_name
        else:
            eprv2_region_name = region_name
        resolver_builtins = self.compute_endpoint_resolver_builtin_defaults(
            region_name=eprv2_region_name,
            service_name=service_name_raw,
            s3_config=s3_config_raw,
            endpoint_bridge=endpoint_bridge,
            client_endpoint_url=endpoint_url,
            legacy_endpoint_url=endpoint.host,
        )
        # botocore does not support client context parameters generically
        # for every service. Instead, the s3 config section entries are
        # available as client context parameters. In the future, endpoint
        # rulesets of services other than s3/s3control may require client
        # context parameters.
        client_context = (
            s3_config_raw if self._is_s3_service(service_name_raw) else {}
        )
        sig_version = (
            client_config.signature_version
            if client_config is not None
            else None
        )
        return AioEndpointRulesetResolver(
            endpoint_ruleset_data=endpoints_ruleset_data,
            partition_data=partition_data,
            service_model=service_model,
            builtins=resolver_builtins,
            client_context=client_context,
            event_emitter=event_emitter,
            use_ssl=is_secure,
            requested_auth_scheme=sig_version,
        )
