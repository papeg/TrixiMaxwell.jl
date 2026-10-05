using TrixiTest

macro test_trixi_include(expr, args...)
    local add_to_additional_ignore_content = [
        # Ignore deprecation warnings from OrdinaryDiffEq
        r"┌ Warning: Passing `stage_limiter!` to the algorithm constructor is deprecated; pass `stage_limiter` as a keyword argument to `solve`/`init` instead\.\n│   caller = .+\n└ @ Core .+\n",
        r"WARNING: Method definition .* in module .* at .* overwritten .*.\n"
    ]
    args = append_to_kwargs(args, :additional_ignore_content,
                            add_to_additional_ignore_content)
    ex = quote
        @test_trixi_include_base($expr, $(args...))
    end
    return esc(ex)
end
