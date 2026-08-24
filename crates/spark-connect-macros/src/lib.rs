//! Procedural macros for the Rust Spark Connect client.
//!
//! Currently provides [`macro@spark_wasm_udf`], which turns a plain Rust
//! function into a Spark UDF that runs on the executors via WebAssembly.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, FnArg, ItemFn, ReturnType, Type};

/// Turn a plain Rust function into a Spark UDF backed by WebAssembly.
///
/// Applied to a numeric scalar function, it:
///
/// * keeps the function callable natively (renamed internally),
/// * emits a `#[no_mangle] pub extern "C"` export **when compiled for
///   `wasm32`**, so the compiled module exports the function under its own
///   name, and
/// * generates a `<name>_udf(module: &[u8]) -> UserDefinedFunction` constructor
///   (on non-`wasm32` targets) with the argument/return types **inferred** from
///   the signature, so callers write only the function.
///
/// ```ignore
/// use spark_connect_macros::spark_wasm_udf;
///
/// #[spark_wasm_udf]
/// fn add_one(x: i64) -> i64 { x + 1 }
///
/// // elsewhere (host build), `module` is the compiled `.wasm` bytes:
/// let f = add_one_udf(module);              // signature auto-inferred
/// df.select(vec![f.call(vec![col("id")])?]);
/// ```
///
/// Supported parameter/return types (this prototype): `i32`, `i64`, `f32`,
/// `f64`. Anything else is a compile error.
#[proc_macro_attribute]
pub fn spark_wasm_udf(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    match expand(func) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// One numeric type, mapped to both the WASM value type and the Spark type.
struct NumericType {
    wasm_valtype: proc_macro2::TokenStream,
    spark_type: proc_macro2::TokenStream,
}

fn map_numeric(ty: &Type) -> Result<NumericType, syn::Error> {
    let ident = match ty {
        Type::Path(p) if p.qself.is_none() => p
            .path
            .get_ident()
            .map(|i| i.to_string())
            .ok_or_else(|| syn::Error::new_spanned(ty, "expected a plain numeric type"))?,
        _ => {
            return Err(syn::Error::new_spanned(
                ty,
                "expected a numeric scalar type (i32, i64, f32, f64)",
            ))
        }
    };
    let (w, s) = match ident.as_str() {
        "i32" => ("I32", quote!(Integer)),
        "i64" => ("I64", quote!(Long)),
        "f32" => ("F32", quote!(Float)),
        "f64" => ("F64", quote!(Double)),
        other => {
            return Err(syn::Error::new_spanned(
                ty,
                format!("#[spark_wasm_udf] supports only i32/i64/f32/f64 (got `{other}`)"),
            ))
        }
    };
    let w = format_ident!("{}", w);
    Ok(NumericType {
        wasm_valtype: quote!(::spark_connect::wasm_udf::WasmValType::#w),
        spark_type: quote!(::spark_connect::types::DataType::#s),
    })
}

fn expand(func: ItemFn) -> Result<proc_macro2::TokenStream, syn::Error> {
    let vis = &func.vis;
    let name = func.sig.ident.clone();
    let udf_name = name.to_string();
    let impl_name = format_ident!("__wasmudf_impl_{}", name);
    let ctor_name = format_ident!("{}_udf", name);

    // Infer argument types.
    let mut arg_valtypes = Vec::new();
    let mut wrapper_params = Vec::new();
    let mut forward_args = Vec::new();
    for (i, input) in func.sig.inputs.iter().enumerate() {
        let ty = match input {
            FnArg::Typed(pt) => &*pt.ty,
            FnArg::Receiver(r) => {
                return Err(syn::Error::new_spanned(
                    r,
                    "#[spark_wasm_udf] cannot be applied to methods (`self`)",
                ))
            }
        };
        // Reject non-numeric early with a clear message.
        let nt = map_numeric(ty)?;
        arg_valtypes.push(nt.wasm_valtype);
        // Use fresh, pattern-free parameter names in the export wrapper so
        // user patterns like `mut a` don't leak into the signature.
        let p = format_ident!("arg{}", i);
        wrapper_params.push(quote!(#p: #ty));
        forward_args.push(quote!(#p));
    }
    if arg_valtypes.is_empty() {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[spark_wasm_udf] requires at least one argument",
        ));
    }

    // Infer return type.
    let ret_ty = match &func.sig.output {
        ReturnType::Type(_, ty) => (**ty).clone(),
        ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                &func.sig,
                "#[spark_wasm_udf] requires a return type",
            ))
        }
    };
    let ret = map_numeric(&ret_ty)?;
    let ret_valtype = ret.wasm_valtype;
    let ret_spark = ret.spark_type;

    // Reject async / generics that would not survive the C ABI.
    if func.sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[spark_wasm_udf] cannot be applied to async functions",
        ));
    }
    if !func.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &func.sig.generics,
            "#[spark_wasm_udf] cannot be applied to generic functions",
        ));
    }

    // The user's function body, kept callable natively under a private name.
    let mut impl_fn = func.clone();
    impl_fn.sig.ident = impl_name.clone();
    impl_fn.vis = syn::Visibility::Inherited;

    let doc = format!(
        "WASM-exported entrypoint for the `{udf_name}` UDF (generated by \
         #[spark_wasm_udf])."
    );

    Ok(quote! {
        #[allow(dead_code)]
        #impl_fn

        // Exported from the compiled `.wasm` under `#udf_name`; a normal
        // `extern "C"` fn otherwise.
        #[doc = #doc]
        #[cfg_attr(target_arch = "wasm32", no_mangle)]
        #vis extern "C" fn #name(#(#wrapper_params),*) -> #ret_ty {
            #impl_name(#(#forward_args),*)
        }

        // Host-side constructor with the signature inferred from the function.
        #[cfg(not(target_arch = "wasm32"))]
        #vis fn #ctor_name(
            module: &[u8],
        ) -> ::spark_connect::wasm_udf::UserDefinedFunction {
            ::spark_connect::wasm_udf::udf(
                #udf_name,
                module.to_vec(),
                #udf_name,
                ::std::vec![#(#arg_valtypes),*],
                #ret_valtype,
                #ret_spark,
            )
        }
    })
}
