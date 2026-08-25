//! Procedural macros for the Rust Spark Connect client.
//!
//! Provides [`macro@spark_wasm_udf`], which turns a module of plain Rust
//! functions into Spark UDFs that run on the executors via WebAssembly.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, FnArg, GenericArgument, Item, ItemFn, ItemMod, PathArguments, ReturnType,
    Type,
};

/// Turn a module of plain Rust functions into Spark UDFs backed by WebAssembly.
///
/// Applied to a `mod`, for each free function inside it the macro:
///
/// * keeps the function callable natively,
/// * emits (when compiled for `wasm32`) a `#[no_mangle] extern "C"` export that
///   decodes the arguments and encodes the result with the length-prefixed
///   binary ABI (`spark_connect::wasm_udf::AbiType`), and
/// * generates, on non-`wasm32` targets, a direct-call constructor
///   `udf::<name>(col0, col1, ...) -> Result<Column>` (one column per argument,
///   arity checked at compile time) plus a builder `udf::<name>_udf() ->
///   UserDefinedFunction` for advanced configuration — both with the Spark
///   signature **inferred** from the Rust signature and the crate's own compiled
///   wasm module embedded.
///
/// ```ignore
/// use spark_connect_macros::spark_wasm_udf;
///
/// #[spark_wasm_udf]
/// mod udfs {
///     pub fn add_one(x: i64) -> i64 { x + 1 }
///     pub fn shout(s: String) -> String { format!("{}!", s.to_uppercase()) }
/// }
///
/// // callers (build.rs embeds the module):
/// df.select(vec![udf::add_one(col("id"))?]);              // direct call
/// df.select(vec![udf::add_one_udf().as_nondeterministic().call(vec![col("id")])?]);
/// ```
///
/// Supported types: `i32`, `i64`, `f32`, `f64`, `bool`, `String`, `Vec<u8>`
/// (binary), `Vec<T>` (array), and `Option<T>` (nullable), nested arbitrarily.
#[proc_macro_attribute]
pub fn spark_wasm_udf(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let module = parse_macro_input!(item as ItemMod);
    match expand(module) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Map a Rust type to a `spark_connect::wasm_udf::AbiType` constructor expr.
fn abi_type(ty: &Type) -> Result<proc_macro2::TokenStream, syn::Error> {
    let root = quote!(::spark_connect::wasm_udf::AbiType);
    if let Some(ident) = path_ident(ty) {
        match ident.as_str() {
            "i32" => return Ok(quote!(#root::I32)),
            "i64" => return Ok(quote!(#root::I64)),
            "f32" => return Ok(quote!(#root::F32)),
            "f64" => return Ok(quote!(#root::F64)),
            "bool" => return Ok(quote!(#root::Bool)),
            "String" => return Ok(quote!(#root::Str)),
            _ => {}
        }
    }
    if let Some(inner) = generic_inner(ty, "Vec") {
        if path_ident(inner).as_deref() == Some("u8") {
            return Ok(quote!(#root::Binary));
        }
        let inner = abi_type(inner)?;
        return Ok(quote!(#root::Array(::std::boxed::Box::new(#inner))));
    }
    if let Some(inner) = generic_inner(ty, "Option") {
        let inner = abi_type(inner)?;
        return Ok(quote!(#root::Nullable(::std::boxed::Box::new(#inner))));
    }
    Err(syn::Error::new_spanned(
        ty,
        "#[spark_wasm_udf] supports i32/i64/f32/f64/bool/String/Vec<u8>/Vec<T>/Option<T> \
         (use owned types, e.g. `String` not `&str`)",
    ))
}

/// The last path segment ident of a plain type path (`i64`, `String`, ...).
fn path_ident(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(p) if p.qself.is_none() => p.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

/// If `ty` is `Wrapper<Inner>` (e.g. `Vec<T>`, `Option<T>`), return `Inner`.
fn generic_inner<'a>(ty: &'a Type, wrapper: &str) -> Option<&'a Type> {
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != wrapper {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    args.args.iter().find_map(|a| match a {
        GenericArgument::Type(t) => Some(t),
        _ => None,
    })
}

struct Udf {
    name: syn::Ident,
    export: syn::Ident,
    arg_tys: Vec<Type>,
    ret_ty: Type,
}

fn collect_udf(func: &ItemFn) -> Result<Udf, syn::Error> {
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
    let name = func.sig.ident.clone();
    let mut arg_tys = Vec::new();
    for input in &func.sig.inputs {
        match input {
            FnArg::Typed(pt) => arg_tys.push((*pt.ty).clone()),
            FnArg::Receiver(r) => {
                return Err(syn::Error::new_spanned(
                    r,
                    "#[spark_wasm_udf] cannot be applied to methods (`self`)",
                ))
            }
        }
    }
    if arg_tys.is_empty() {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[spark_wasm_udf] requires at least one argument",
        ));
    }
    let ret_ty = match &func.sig.output {
        ReturnType::Type(_, ty) => (**ty).clone(),
        ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                &func.sig,
                "#[spark_wasm_udf] requires a return type",
            ))
        }
    };
    let export = format_ident!("spark_udf_{}", name);
    Ok(Udf {
        name,
        export,
        arg_tys,
        ret_ty,
    })
}

fn expand(mut module: ItemMod) -> Result<proc_macro2::TokenStream, syn::Error> {
    let mod_ident = module.ident.clone();
    let Some((_, items)) = &module.content else {
        return Err(syn::Error::new_spanned(
            &module,
            "#[spark_wasm_udf] must be applied to an inline module with a body",
        ));
    };

    let udfs: Vec<Udf> = items
        .iter()
        .filter_map(|it| match it {
            Item::Fn(f) => Some(collect_udf(f)),
            _ => None,
        })
        .collect::<Result<_, _>>()?;
    if udfs.is_empty() {
        return Err(syn::Error::new_spanned(
            &module,
            "#[spark_wasm_udf] module contains no functions",
        ));
    }

    // WASM export wrappers, injected into the module (compiled only for wasm32).
    let mut wrappers = Vec::new();
    for u in &udfs {
        let Udf {
            name,
            export,
            arg_tys,
            ret_ty,
        } = u;
        let params: Vec<_> = arg_tys
            .iter()
            .map(|ty| quote!(<#ty as crate::spark_wasm_rt::Abi>::decode(&mut __r)))
            .collect();
        wrappers.push(quote! {
            #[cfg(target_arch = "wasm32")]
            #[no_mangle]
            pub extern "C" fn #export(__ptr: *const u8, __len: u32) -> u64 {
                let __b = unsafe { ::core::slice::from_raw_parts(__ptr, __len as usize) };
                let mut __r = crate::spark_wasm_rt::Reader::new(__b);
                let __out: #ret_ty = #name(#(#params),*);
                let mut __o = ::std::vec::Vec::new();
                crate::spark_wasm_rt::Abi::encode(&__out, &mut __o);
                crate::spark_wasm_rt::finish(__o)
            }
        });
    }
    if let Some((_, content)) = &mut module.content {
        for w in wrappers {
            content.push(syn::parse2(w)?);
        }
    }

    // Host-side constructors, grouped under `udf::`. For each UDF the macro
    // emits two functions:
    //   * `udf::<name>(col0, col1, ...) -> Result<Column>` — the direct call,
    //     one column argument per Rust parameter (arity checked at compile
    //     time), mirroring calling a PySpark UDF on columns.
    //   * `udf::<name>_udf() -> UserDefinedFunction` — the builder, for advanced
    //     configuration (`.as_nondeterministic()`, `.with_packer()`, ...) before
    //     `.call(cols)`.
    let mut ctors = Vec::new();
    for u in &udfs {
        let Udf {
            name,
            export,
            arg_tys,
            ret_ty,
        } = u;
        let name_str = name.to_string();
        let export_str = export.to_string();
        let builder = format_ident!("{}_udf", name);
        let arg_abis: Vec<_> = arg_tys.iter().map(abi_type).collect::<Result<_, _>>()?;
        let ret_abi = abi_type(ret_ty)?;
        let params: Vec<syn::Ident> = (0..arg_tys.len())
            .map(|i| format_ident!("__a{i}"))
            .collect();
        let builder_doc = format!(
            "Builder for the `{name_str}` UDF, generated by `#[spark_wasm_udf]`. \
             For advanced configuration (`.as_nondeterministic()`, `.with_packer()`) \
             before `.call(cols)`; the common case is `{name_str}(cols...)`."
        );
        let call_doc = format!(
            "Apply the `{name_str}` UDF to columns, generated by `#[spark_wasm_udf]`. \
             Takes one column per argument and returns the result column."
        );
        ctors.push(quote! {
            #[doc = #builder_doc]
            pub fn #builder() -> ::spark_connect::wasm_udf::UserDefinedFunction {
                const __MODULE: &[u8] = include_bytes!(env!("WASM_UDFS_MODULE"));
                ::spark_connect::wasm_udf::udf(
                    #name_str,
                    __MODULE.to_vec(),
                    #export_str,
                    ::std::vec![#(#arg_abis),*],
                    #ret_abi,
                )
            }

            #[doc = #call_doc]
            pub fn #name(
                #(#params: ::spark_connect::Column),*
            ) -> ::spark_connect::Result<::spark_connect::Column> {
                #builder().call(::std::vec![#(#params),*])
            }
        });
    }

    let doc =
        format!("UDF constructors generated by `#[spark_wasm_udf]` from module `{mod_ident}`.");
    Ok(quote! {
        #module

        #[cfg(not(target_arch = "wasm32"))]
        #[doc = #doc]
        pub mod udf {
            #(#ctors)*
        }
    })
}
