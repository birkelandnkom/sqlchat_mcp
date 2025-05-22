import chainlit as cl
from openai import AsyncAzureOpenAI
import json
import os
from dotenv import load_dotenv

try:
    from mcp import ClientSession
    MCP_CLASSES_AVAILABLE = True
    print("DEBUG: Successfully imported ClientSession from 'mcp'")
except ImportError:
    ClientSession = None 
    MCP_CLASSES_AVAILABLE = False
    print("DEBUG: Warning: Could not import ClientSession from 'mcp'.")
    print("DEBUG: Please ensure your Chainlit environment provides this, or check for 'mcp-sdk' installation if it's a separate requirement.")


load_dotenv()

azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
azure_deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
postgres_connection_string = os.environ.get("POSTGRESQL_CONNECTION_STRING")


if not all([azure_api_key, azure_endpoint, azure_api_version, azure_deployment_name]):
    print("DEBUG: Warning: Azure OpenAI environment variables are not fully set.")
    azure_client = None
else:
    azure_client = AsyncAzureOpenAI(
       api_key=azure_api_key,
       azure_endpoint=azure_endpoint,
       api_version=azure_api_version
    )
    print(f"DEBUG: Azure OpenAI client initialized for deployment: {azure_deployment_name}, endpoint: {azure_endpoint}")


POSTGRES_MCP_SERVER_NAME = "postgres"

@cl.on_chat_start
async def start_chat():
    print("DEBUG: on_chat_start called")
    if not MCP_CLASSES_AVAILABLE:
        await cl.Message(content="Critical Error: MCP ClientSession class not available. Please check imports and dependencies.").send()
    if not azure_client:
        await cl.Message(content="Critical Error: Azure OpenAI client not configured. Please check environment variables.").send()
        
    await cl.Message(content="Hello! I can help you query your PostgreSQL database. What would you like to know?").send()
    cl.user_session.set("history", [])
    cl.user_session.set("openai_tools_for_mcp", {})
    print("DEBUG: Chat session initialized.")

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    connection_name = getattr(connection, 'name', 'unknown_mcp_server')
    print(f"DEBUG: on_mcp_connect called for connection: {connection_name}")

    if connection_name == POSTGRES_MCP_SERVER_NAME:
        print(f"DEBUG: ***** SUCCESSFULLY ENTERED on_mcp_connect FOR {POSTGRES_MCP_SERVER_NAME} *****")
        await cl.Message(content=f"Chainlit successfully connected to MCP server: '{POSTGRES_MCP_SERVER_NAME}'! Session object: {session}").send()
        # Now, let's try listing tools here and see what happens
        try:
            list_tools_result = await session.list_tools()
            print(f"DEBUG: list_tools_result for {connection_name} inside specific block: {list_tools_result}")
            if list_tools_result and hasattr(list_tools_result, 'tools') and list_tools_result.tools:
                tool_names = [t.name for t in list_tools_result.tools]
                await cl.Message(content=f"Tools found for '{connection_name}': {tool_names}").send()
                 # Try to populate the session tools here directly
                current_mcp_tools_formatted = cl.user_session.get("openai_tools_for_mcp", {})
                connection_openai_tools = []
                for tool_spec in list_tools_result.tools:
                    openai_tool_name = f"{connection_name}__{tool_spec.name.replace('/', '__')}"
                    connection_openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": openai_tool_name,
                            "description": tool_spec.description or "",
                            "parameters": tool_spec.inputSchema or {"type": "object", "properties": {}}
                        }
                    })
                current_mcp_tools_formatted[connection_name] = connection_openai_tools
                cl.user_session.set("openai_tools_for_mcp", current_mcp_tools_formatted)
                print(f"DEBUG: Tools for {connection_name} set in session: {connection_openai_tools}")

            else:
                await cl.Message(content=f"No tools returned by list_tools() for '{connection_name}' or result format unexpected.").send()

        except Exception as e:
            print(f"DEBUG: Error calling list_tools for '{connection_name}': {str(e)}")
            await cl.Message(content=f"Error calling list_tools for '{connection_name}': {str(e)}").send()
    else:
        print(f"DEBUG: on_mcp_connect called for a different server: {connection_name}")
        await cl.Message(content=f"Error fetching/processing tools from MCP server '{connection_name}': {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    print(f"DEBUG: on_message received: {message.content}")
    if not MCP_CLASSES_AVAILABLE or ClientSession is None:
        await cl.Message(content="Error: MCP ClientSession class not available. Cannot process message.").send()
        return
    if not azure_client:
        await cl.Message(content="Error: Azure OpenAI client not configured. Cannot process message.").send()
        return

    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    messages_for_llm = [
        {"role": "system", "content": "You are a helpful assistant that can query a PostgreSQL database. When you need to query the database, use the tool provided by the 'postgres' MCP server. The SQL should be valid PostgreSQL."}
    ]
    messages_for_llm.extend(history)

    all_openai_mcp_tools_dict = cl.user_session.get("openai_tools_for_mcp", {})
    active_openai_tools = all_openai_mcp_tools_dict.get(POSTGRES_MCP_SERVER_NAME, [])
    # This is the critical point: if active_openai_tools is empty, the LLM won't know about the tools.
    print(f"DEBUG: Retrieved 'openai_tools_for_mcp' from session for '{POSTGRES_MCP_SERVER_NAME}': {active_openai_tools}")
    
    settings = {
        "model": azure_deployment_name,
        "messages": messages_for_llm,
        "tools": active_openai_tools if active_openai_tools else None,
        "tool_choice": "auto" if active_openai_tools else None,
    }

    response_message_content = ""
    
    cl_msg = cl.Message(content="")
    await cl_msg.send() 

    try:
        print(f"DEBUG: Calling Azure OpenAI with settings: {json.dumps(settings, indent=2)}")
        stream = await azure_client.chat.completions.create(**settings, stream=True)
        
        full_response_text = ""
        tool_calls_data = [] 

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0: 
                delta = chunk.choices[0].delta
                content_chunk = delta.content
                tool_call_chunks = delta.tool_calls

                if content_chunk:
                    full_response_text += content_chunk
                    await cl_msg.stream_token(content_chunk)

                if tool_call_chunks:
                    print(f"DEBUG: Received tool_call_chunks: {tool_call_chunks}")
                    for tc_chunk in tool_call_chunks:
                        while tc_chunk.index >= len(tool_calls_data):
                            tool_calls_data.append({})
                        
                        current_tool_call = tool_calls_data[tc_chunk.index]
                        
                        if "id" not in current_tool_call and tc_chunk.id:
                                current_tool_call["id"] = tc_chunk.id
                        if "type" not in current_tool_call and tc_chunk.type:
                                current_tool_call["type"] = tc_chunk.type
                        elif "type" not in current_tool_call:
                                current_tool_call["type"] = "function"
                        
                        if "function" not in current_tool_call:
                                current_tool_call["function"] = {"name": "", "arguments": ""}

                        if tc_chunk.id:
                            current_tool_call["id"] = tc_chunk.id

                        if tc_chunk.function:
                            if tc_chunk.function.name:
                                current_tool_call["function"]["name"] += tc_chunk.function.name
                            if tc_chunk.function.arguments:
                                current_tool_call["function"]["arguments"] += tc_chunk.function.arguments
            else:
                pass
        
        print(f"DEBUG: LLM initial response text: '{full_response_text}'")
        print(f"DEBUG: LLM tool_calls_data: {tool_calls_data}")

        if tool_calls_data and any(tc.get("function", {}).get("name") for tc in tool_calls_data):
            await cl_msg.stream_token("\n\nUsing tool(s)...\n")
            
            assistant_message_for_history = {"role": "assistant", "tool_calls": tool_calls_data}
            if full_response_text:
                assistant_message_for_history["content"] = full_response_text
            
            history.append(assistant_message_for_history)
            messages_for_llm.append(assistant_message_for_history)

            mcp_sessions = cl.context.session.mcp_sessions
            print(f"DEBUG: Available MCP sessions: {mcp_sessions.keys() if mcp_sessions else 'None'}")
            postgres_mcp_session_tuple = mcp_sessions.get(POSTGRES_MCP_SERVER_NAME) if mcp_sessions else None
            
            if not postgres_mcp_session_tuple:
                error_msg_no_session = f"Error: Could not find active MCP session for '{POSTGRES_MCP_SERVER_NAME}'."
                print(f"DEBUG: {error_msg_no_session}")
                await cl_msg.stream_token(error_msg_no_session + "\n")
                await cl.Message(content=error_msg_no_session).send() 
                return
            
            actual_mcp_session: ClientSession = postgres_mcp_session_tuple[0]
            print(f"DEBUG: Obtained MCP session for '{POSTGRES_MCP_SERVER_NAME}'")

            for tool_call in tool_calls_data:
                if not tool_call.get("function", {}).get("name"):
                    continue

                openai_tool_name = tool_call["function"]["name"]
                print(f"DEBUG: LLM wants to call OpenAI tool: {openai_tool_name}")
                
                expected_prefix = POSTGRES_MCP_SERVER_NAME + "__"
                if openai_tool_name.startswith(expected_prefix):
                    mcp_tool_name_actual = openai_tool_name[len(expected_prefix):].replace("__", "/")
                else:
                    error_msg_tool_format = f"Error: Tool name '{openai_tool_name}' not in expected format for server '{POSTGRES_MCP_SERVER_NAME}'."
                    print(f"DEBUG: {error_msg_tool_format}")
                    await cl_msg.stream_token(error_msg_tool_format + "\n")
                    messages_for_llm.append({
                        "tool_call_id": tool_call["id"], "role": "tool", "name": openai_tool_name,
                        "content": error_msg_tool_format
                    })
                    continue
                
                tool_arguments_str = tool_call["function"].get("arguments", "{}")
                try:
                    tool_args = json.loads(tool_arguments_str)
                    print(f"DEBUG: Parsed tool arguments for {mcp_tool_name_actual}: {tool_args}")
                except json.JSONDecodeError as json_err:
                    error_content = f"Error: Could not parse arguments for tool {mcp_tool_name_actual}: {tool_arguments_str}. JSON Error: {json_err}"
                    print(f"DEBUG: {error_content}")
                    await cl_msg.stream_token(error_content + "\n")
                    messages_for_llm.append({
                        "tool_call_id": tool_call["id"], "role": "tool", "name": openai_tool_name,
                        "content": f"Error: Invalid JSON arguments for tool {mcp_tool_name_actual}. Arguments: {tool_arguments_str}",
                    })
                    continue

                await cl_msg.stream_token(f"Calling MCP tool: `{POSTGRES_MCP_SERVER_NAME}/{mcp_tool_name_actual}` with args: `{tool_args}`\n")

                try:
                    print(f"DEBUG: Calling actual_mcp_session.call_tool for {mcp_tool_name_actual} with args {tool_args}")
                    tool_response_mcp_sdk = await actual_mcp_session.call_tool(name=mcp_tool_name_actual, arguments=tool_args)
                    print(f"DEBUG: Raw response from MCP server: {tool_response_mcp_sdk}")
                    
                    tool_output_text = "Tool execution failed or returned no text."
                    if tool_response_mcp_sdk and tool_response_mcp_sdk.content:
                        for item in tool_response_mcp_sdk.content:
                            if hasattr(item, 'text') and item.type == "text":
                                tool_output_text = item.text
                                break
                    print(f"DEBUG: Extracted tool_output_text: {tool_output_text}")
                    
                    await cl_msg.stream_token(f"Tool response for `{mcp_tool_name_actual}`: \n```json\n{tool_output_text}\n```\n")

                    messages_for_llm.append({
                        "tool_call_id": tool_call["id"], "role": "tool", "name": openai_tool_name,
                        "content": tool_output_text,
                    })
                except Exception as tool_call_e:
                    error_str = f"Error calling MCP tool {mcp_tool_name_actual}: {str(tool_call_e)}"
                    print(f"DEBUG: {error_str}")
                    await cl_msg.stream_token(error_str + "\n")
                    messages_for_llm.append({
                        "tool_call_id": tool_call["id"], "role": "tool", "name": openai_tool_name,
                        "content": error_str,
                    })
            
            print(f"DEBUG: Calling Azure OpenAI again with tool responses: {json.dumps(messages_for_llm, indent=2)}")
            stream_after_tool_call = await azure_client.chat.completions.create(
                model=settings["model"],
                messages=messages_for_llm,
                stream=True
            )
            final_llm_response_text = ""
            async for chunk_after_tool in stream_after_tool_call: 
                if chunk_after_tool.choices and len(chunk_after_tool.choices) > 0: 
                    content_chunk_after_tool = chunk_after_tool.choices[0].delta.content 
                    if content_chunk_after_tool:
                        final_llm_response_text += content_chunk_after_tool
                        await cl_msg.stream_token(content_chunk_after_tool)
            
            response_message_content = final_llm_response_text
            print(f"DEBUG: Final LLM response after tool call: '{response_message_content}'")
        
        else: 
            response_message_content = full_response_text
            print(f"DEBUG: LLM response (no tool call): '{response_message_content}'")


        if cl_msg.content != response_message_content and response_message_content:
             await cl_msg.set_content(response_message_content)
        elif not cl_msg.content and response_message_content: 
             await cl_msg.set_content(response_message_content)
        await cl_msg.update()


    except Exception as e:
        print(f"DEBUG: Error in main on_message handler: {str(e)}")
        response_message_content = f"An error occurred: {str(e)}"
        if cl_msg.streaming: 
            await cl_msg.set_content(response_message_content)
            await cl_msg.update()
        else: 
            await cl.Message(content=response_message_content).send()


    if response_message_content:
      history.append({"role": "assistant", "content": response_message_content})
    cl.user_session.set("history", history)

