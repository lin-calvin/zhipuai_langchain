import warnings
warnings.warn("This library is depracted, use langchain_community.chat_models.ChatZhipuAI insted", DeprecationWarning)
from  langchain_community.chat_models import ChatZhipuAI as ZhiPuAIChatModel
# """ChatModel wrapper which returns user input as the response.."""
# import asyncio
# from functools import partial
# from io import StringIO
# import json
# from typing import Any, Callable, Dict, List, Mapping, Optional
#
#
# from langchain.callbacks.manager import (
#     AsyncCallbackManagerForLLMRun,
#     CallbackManagerForLLMRun,
# )
# from langchain.chat_models.base import BaseChatModel
# from langchain.llms.utils import enforce_stop_tokens
# from langchain.pydantic_v1 import Field
# from langchain.schema.messages import (
#     BaseMessage,
#     HumanMessage,
#     AIMessage,
#     _message_from_dict,
#     messages_to_dict,
# )
# from langchain.schema.output import ChatGeneration, ChatResult
# import zhipuai
# from langchain.pydantic_v1 import BaseModel
# from langchain.schema.embeddings import Embeddings
#
# class ZhiPuAiEmbeddings(Embeddings, BaseModel):
#     api_key:str = Field()
#     def __init__(self,*args,**kwargs):
#         super().__init__(*args,**kwargs)
#         try:
#             import zhipuai
#             zhipuai.api_key=self.api_key
#         except ImportError:
#             raise RuntimeError("Zhipuai hasn't been installed, install it via 'pip install zhipuai'")
#
#     def embed_query(self, text: str) -> List[float]:
#         return zhipuai.model_api.invoke(
#             model="text_embedding",
#             prompt=text
#         )['data']['embedding']
#
#     def embed_documents(self, texts: List[str])-> List[List[float]]:
#         return list(map(self.embed_query,texts))
#
# class ZhiPuAIChatModel(BaseChatModel):
#     """ChatModel which returns user input as the response."""
#
#     api_key:str = Field()
#     model:str=Field()
#     top_p:int=Field()
#     temperature:int=Field()
#     @property
#     def _identifying_params(self) -> Dict[str, Any]:
#         return {"model_name":self.model}
#     @property
#     def _llm_type(self) -> str:
#         """Returns the type of LLM."""
#         return self.model
#     def __init__(self,*args,**kwargs):
#         super().__init__(*args,**kwargs)
#         try:
#             import zhipuai
#             zhipuai.api_key=self.api_key
#         except ImportError:
#             raise RuntimeError("Zhipuai hasn't been installed, install it via 'pip install zhipuai'")
#     def _generate(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         """
#         Displays the messages to the user and returns their input as a response.
#
#         Args:
#             messages (List[BaseMessage]): The messages to be displayed to the user.
#             stop (Optional[List[str]]): A list of stop strings.
#             run_manager (Optional[CallbackManagerForLLMRun]): Currently not used.
#
#         Returns:
#             ChatResult: The user's input as a response.
#         """
#         prompt=[]
#         if type(messages)==str:
#             prompt= [{"role": "user", "content": messages}]
#         else:
#             #print(messages)
#             prompt=[{"role":['assistant','user'][type(message)==HumanMessage],"content": message.content} for message in messages]
#             prompt.insert(0,{"role":"user","content":""})#This is a small hack but workable
#         response = zhipuai.model_api.invoke(
#             model=self.model,
#             prompt=prompt,
#             top_p=self.top_p,
#             temperature=self.temperature,
#         )
#         if response['code']!=200:
#             raise RuntimeError(response)
#         response=response['data']['choices'][0]['content']
#         response=json.loads(response)
#         return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])
#
#     async def _agenerate(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         func = partial(
#             self._generate, messages, stop=stop, run_manager=run_manager, **kwargs
#         )
#         return await asyncio.get_event_loop().run_in_executor(None, func)
#
#
