import asyncio

import click

from src.llm_client.client import LLMClient
from src.llm_client.schema import LLMResponse, Message, Provider


@click.group()
def cli():
    """LLM Client CLI — quickly test providers."""


@cli.command()
@click.argument("prompt")
@click.option(
    "--provider", "-p", default=None, type=click.Choice([p.value for p in Provider])
)
@click.option("--system", "-s", default=None, help="System prompt")
@click.option("--stream", "do_stream", is_flag=True, default=False)
@click.option("--max-tokens", default=1024, type=int)
@click.option("--temperature", default=0.7, type=float)
def ask(
    prompt: str,
    provider: str | None,
    system: str | None,
    do_stream: bool,
    max_tokens: int,
    temperature: float,
):
    """Send a prompt and print the response."""
    provider_enum = Provider(provider) if provider else None
    messages = [Message(role="user", content=prompt)]
    llm = LLMClient()

    async def run():
        if do_stream:
            async for chunk in llm.stream(
                messages, system=system, provider=provider_enum, max_tokens=max_tokens
            ):
                print(chunk, end="", flush=True)
            print()
        else:
            response = await llm.complete(
                messages,
                system=system,
                provider=provider_enum,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            click.echo(response.content)
            click.echo(
                f"\n[{response.provider.value}/{response.model}] "
                f"{response.usage.total_tokens} tokens | "
                f"{response.latency_ms:.0f}ms"
            )

    asyncio.run(run())


@cli.command()
@click.argument("prompt")
def compare(prompt: str):
    """Run the same prompt against all configured providers."""
    messages = [Message(role="user", content=prompt)]
    llm = LLMClient()

    async def run():
        results = await llm.compare(messages)
        for provider, response in results.items():
            click.echo(f"\n{'=' * 50}")
            click.echo(f"Provider: {provider.value}")
            if isinstance(response, LLMResponse):
                click.echo(response.content)
                click.echo(
                    f"[{response.model}] {response.usage.total_tokens} tokens | {response.latency_ms:.0f}ms"
                )
            else:
                click.echo(f"ERROR: {response}")

    asyncio.run(run())


if __name__ == "__main__":
    cli()