"""Generated client library for iam version v2alpha."""
# NOTE: This file is autogenerated and should not be edited by hand.

from __future__ import absolute_import

from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v2alpha import iam_v2alpha_messages as messages


class IamV2alpha(base_api.BaseApiClient):
  """Generated client library for service iam version v2alpha."""

  MESSAGES_MODULE = messages
  BASE_URL = 'https://iam.googleapis.com/'
  MTLS_BASE_URL = 'https://iam.mtls.googleapis.com/'

  _PACKAGE = 'iam'
  _SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
  _VERSION = 'v2alpha'
  _CLIENT_ID = 'CLIENT_ID'
  _CLIENT_SECRET = 'CLIENT_SECRET'
  _USER_AGENT = 'google-cloud-sdk'
  _CLIENT_CLASS_NAME = 'IamV2alpha'
  _URL_VERSION = 'v2alpha'
  _API_KEY = None

  def __init__(self, url='', credentials=None,
               get_credentials=True, http=None, model=None,
               log_request=False, log_response=False,
               credentials_args=None, default_global_params=None,
               additional_http_headers=None, response_encoding=None):
    """Create a new iam handle."""
    url = url or self.BASE_URL
    super(IamV2alpha, self).__init__(
        url, credentials=credentials,
        get_credentials=get_credentials, http=http, model=model,
        log_request=log_request, log_response=log_response,
        credentials_args=credentials_args,
        default_global_params=default_global_params,
        additional_http_headers=additional_http_headers,
        response_encoding=response_encoding)
    self.policies_operations = self.PoliciesOperationsService(self)
    self.policies = self.PoliciesService(self)

  class PoliciesOperationsService(base_api.BaseApiService):
    """Service class for the policies_operations resource."""

    _NAME = 'policies_operations'

    def __init__(self, client):
      super(IamV2alpha.PoliciesOperationsService, self).__init__(client)
      self._upload_configs = {
          }

    def Get(self, request, global_params=None):
      r"""Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (IamPoliciesOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v2alpha/policies/{policiesId}/{policiesId1}/{policiesId2}/operations/{operationsId}',
        http_method='GET',
        method_id='iam.policies.operations.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v2alpha/{+name}',
        request_field='',
        request_type_name='IamPoliciesOperationsGetRequest',
        response_type_name='GoogleLongrunningOperation',
        supports_download=False,
    )

  class PoliciesService(base_api.BaseApiService):
    """Service class for the policies resource."""

    _NAME = 'policies'

    def __init__(self, client):
      super(IamV2alpha.PoliciesService, self).__init__(client)
      self._upload_configs = {
          }

    def CreatePolicy(self, request, global_params=None):
      r"""Creates a policy. All the policies attached to a specific resource must have unique IDs.

      Args:
        request: (IamPoliciesCreatePolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
      config = self.GetMethodConfig('CreatePolicy')
      return self._RunMethod(
          config, request, global_params=global_params)

    CreatePolicy.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v2alpha/policies/{policiesId}/{policiesId1}',
        http_method='POST',
        method_id='iam.policies.createPolicy',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['policyId'],
        relative_path='v2alpha/{+parent}',
        request_field='googleIamV2alphaPolicy',
        request_type_name='IamPoliciesCreatePolicyRequest',
        response_type_name='GoogleLongrunningOperation',
        supports_download=False,
    )

    def Delete(self, request, global_params=None):
      r"""Deletes a policy. Can provide 'Policy.etag' to enforce delete from last read for optimistic concurrency control.

      Args:
        request: (IamPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
      config = self.GetMethodConfig('Delete')
      return self._RunMethod(
          config, request, global_params=global_params)

    Delete.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v2alpha/policies/{policiesId}/{policiesId1}/{policiesId2}',
        http_method='DELETE',
        method_id='iam.policies.delete',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['etag'],
        relative_path='v2alpha/{+name}',
        request_field='',
        request_type_name='IamPoliciesDeleteRequest',
        response_type_name='GoogleLongrunningOperation',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      r"""Gets a policy.

      Args:
        request: (IamPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV2alphaPolicy) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v2alpha/policies/{policiesId}/{policiesId1}/{policiesId2}',
        http_method='GET',
        method_id='iam.policies.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v2alpha/{+name}',
        request_field='',
        request_type_name='IamPoliciesGetRequest',
        response_type_name='GoogleIamV2alphaPolicy',
        supports_download=False,
    )

    def ListPolicies(self, request, global_params=None):
      r"""Retrieves all of the policies attached to the specified resource, of the given kind. Only policy metadata is listed; specifically `policy.rules` is omitted.

      Args:
        request: (IamPoliciesListPoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV2alphaListPoliciesResponse) The response message.
      """
      config = self.GetMethodConfig('ListPolicies')
      return self._RunMethod(
          config, request, global_params=global_params)

    ListPolicies.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v2alpha/policies/{policiesId}/{policiesId1}',
        http_method='GET',
        method_id='iam.policies.listPolicies',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['pageSize', 'pageToken'],
        relative_path='v2alpha/{+parent}',
        request_field='',
        request_type_name='IamPoliciesListPoliciesRequest',
        response_type_name='GoogleIamV2alphaListPoliciesResponse',
        supports_download=False,
    )

    def Update(self, request, global_params=None):
      r"""Updates the specified existing policy. Only `Policy.rules` and `Policy.display_name` may be updated. Need to provide 'Policy.etag' to enforce update from last read for optimistic concurrency control.

      Args:
        request: (GoogleIamV2alphaPolicy) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
      config = self.GetMethodConfig('Update')
      return self._RunMethod(
          config, request, global_params=global_params)

    Update.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v2alpha/policies/{policiesId}/{policiesId1}/{policiesId2}',
        http_method='PUT',
        method_id='iam.policies.update',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v2alpha/{+name}',
        request_field='<request>',
        request_type_name='GoogleIamV2alphaPolicy',
        response_type_name='GoogleLongrunningOperation',
        supports_download=False,
    )
