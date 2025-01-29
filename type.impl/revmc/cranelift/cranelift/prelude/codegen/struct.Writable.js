(function() {
    var type_impls = Object.fromEntries([["revmc",[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Clone-for-Writable%3CT%3E\" class=\"impl\"><a href=\"#impl-Clone-for-Writable%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/clone/trait.Clone.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::clone::Clone\">Clone</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/clone/trait.Clone.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::clone::Clone\">Clone</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone\" class=\"method trait-impl\"><a href=\"#method.clone\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/clone/trait.Clone.html#tymethod.clone\" class=\"fn\">clone</a>(&amp;self) -&gt; <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;</h4></section></summary><div class='docblock'>Returns a copy of the value. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/clone/trait.Clone.html#tymethod.clone\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone_from\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/clone.rs.html#174\">Source</a></span><a href=\"#method.clone_from\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/clone/trait.Clone.html#method.clone_from\" class=\"fn\">clone_from</a>(&amp;mut self, source: &amp;Self)</h4></section></summary><div class='docblock'>Performs copy-assignment from <code>source</code>. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/clone/trait.Clone.html#method.clone_from\">Read more</a></div></details></div></details>","Clone","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableGpr","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableXmm"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Debug-for-Writable%3CT%3E\" class=\"impl\"><a href=\"#impl-Debug-for-Writable%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/fmt/trait.Debug.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::fmt::Debug\">Debug</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/fmt/trait.Debug.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::fmt::Debug\">Debug</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.fmt\" class=\"method trait-impl\"><a href=\"#method.fmt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/fmt/trait.Debug.html#tymethod.fmt\" class=\"fn\">fmt</a>(&amp;self, f: &amp;mut <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/fmt/struct.Formatter.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::entity::__core::fmt::Formatter\">Formatter</a>&lt;'_&gt;) -&gt; <a class=\"enum\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/result/enum.Result.html\" title=\"enum revmc::cranelift::cranelift::prelude::codegen::entity::__core::result::Result\">Result</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.unit.html\">()</a>, <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/fmt/struct.Error.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::entity::__core::fmt::Error\">Error</a>&gt;</h4></section></summary><div class='docblock'>Formats the value using the given formatter. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/fmt/trait.Debug.html#tymethod.fmt\">Read more</a></div></details></div></details>","Debug","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableGpr","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableXmm"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-FromWritableReg-for-Writable%3CGpr%3E\" class=\"impl\"><a href=\"#impl-FromWritableReg-for-Writable%3CGpr%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/trait.FromWritableReg.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::FromWritableReg\">FromWritableReg</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/struct.Gpr.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::Gpr\">Gpr</a>&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.from_writable_reg\" class=\"method trait-impl\"><a href=\"#method.from_writable_reg\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/trait.FromWritableReg.html#tymethod.from_writable_reg\" class=\"fn\">from_writable_reg</a>(w: <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Reg.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Reg\">Reg</a>&gt;) -&gt; <a class=\"enum\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/option/enum.Option.html\" title=\"enum revmc::cranelift::cranelift::prelude::codegen::entity::__core::option::Option\">Option</a>&lt;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/struct.Gpr.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::Gpr\">Gpr</a>&gt;&gt;</h4></section></summary><div class='docblock'>Convert <code>Writable&lt;Reg&gt;</code> to <code>Writable{Xmm,Gpr}</code>.</div></details></div></details>","FromWritableReg","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableGpr"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-FromWritableReg-for-Writable%3CXmm%3E\" class=\"impl\"><a href=\"#impl-FromWritableReg-for-Writable%3CXmm%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/trait.FromWritableReg.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::FromWritableReg\">FromWritableReg</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/struct.Xmm.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::Xmm\">Xmm</a>&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.from_writable_reg\" class=\"method trait-impl\"><a href=\"#method.from_writable_reg\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/trait.FromWritableReg.html#tymethod.from_writable_reg\" class=\"fn\">from_writable_reg</a>(w: <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Reg.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Reg\">Reg</a>&gt;) -&gt; <a class=\"enum\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/option/enum.Option.html\" title=\"enum revmc::cranelift::cranelift::prelude::codegen::entity::__core::option::Option\">Option</a>&lt;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/struct.Xmm.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::Xmm\">Xmm</a>&gt;&gt;</h4></section></summary><div class='docblock'>Convert <code>Writable&lt;Reg&gt;</code> to <code>Writable{Xmm,Gpr}</code>.</div></details></div></details>","FromWritableReg","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableXmm"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Hash-for-Writable%3CT%3E\" class=\"impl\"><a href=\"#impl-Hash-for-Writable%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/hash/trait.Hash.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::hash::Hash\">Hash</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/hash/trait.Hash.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::hash::Hash\">Hash</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.hash\" class=\"method trait-impl\"><a href=\"#method.hash\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/hash/trait.Hash.html#tymethod.hash\" class=\"fn\">hash</a>&lt;__H&gt;(&amp;self, state: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;mut __H</a>)<div class=\"where\">where\n    __H: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/hash/trait.Hasher.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::hash::Hasher\">Hasher</a>,</div></h4></section></summary><div class='docblock'>Feeds this value into the given <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/hash/trait.Hasher.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::hash::Hasher\"><code>Hasher</code></a>. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/hash/trait.Hash.html#tymethod.hash\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.hash_slice\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.3.0\">1.3.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/hash/mod.rs.html#235-237\">Source</a></span><a href=\"#method.hash_slice\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/hash/trait.Hash.html#method.hash_slice\" class=\"fn\">hash_slice</a>&lt;H&gt;(data: &amp;[Self], state: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;mut H</a>)<div class=\"where\">where\n    H: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/hash/trait.Hasher.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::hash::Hasher\">Hasher</a>,\n    Self: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/marker/trait.Sized.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::marker::Sized\">Sized</a>,</div></h4></section></summary><div class='docblock'>Feeds a slice of this type into the given <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/hash/trait.Hasher.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::hash::Hasher\"><code>Hasher</code></a>. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/hash/trait.Hash.html#method.hash_slice\">Read more</a></div></details></div></details>","Hash","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableGpr","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableXmm"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Ord-for-Writable%3CT%3E\" class=\"impl\"><a href=\"#impl-Ord-for-Writable%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.Ord.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::cmp::Ord\">Ord</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.Ord.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::cmp::Ord\">Ord</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.cmp\" class=\"method trait-impl\"><a href=\"#method.cmp\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.Ord.html#tymethod.cmp\" class=\"fn\">cmp</a>(&amp;self, other: &amp;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;) -&gt; <a class=\"enum\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/enum.Ordering.html\" title=\"enum revmc::cranelift::cranelift::prelude::codegen::entity::__core::cmp::Ordering\">Ordering</a></h4></section></summary><div class='docblock'>This method returns an <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/enum.Ordering.html\" title=\"enum revmc::cranelift::cranelift::prelude::codegen::entity::__core::cmp::Ordering\"><code>Ordering</code></a> between <code>self</code> and <code>other</code>. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.Ord.html#tymethod.cmp\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.max\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.21.0\">1.21.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/cmp.rs.html#980-982\">Source</a></span><a href=\"#method.max\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.Ord.html#method.max\" class=\"fn\">max</a>(self, other: Self) -&gt; Self<div class=\"where\">where\n    Self: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/marker/trait.Sized.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::marker::Sized\">Sized</a>,</div></h4></section></summary><div class='docblock'>Compares and returns the maximum of two values. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.Ord.html#method.max\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.min\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.21.0\">1.21.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/cmp.rs.html#1001-1003\">Source</a></span><a href=\"#method.min\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.Ord.html#method.min\" class=\"fn\">min</a>(self, other: Self) -&gt; Self<div class=\"where\">where\n    Self: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/marker/trait.Sized.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::marker::Sized\">Sized</a>,</div></h4></section></summary><div class='docblock'>Compares and returns the minimum of two values. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.Ord.html#method.min\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.clamp\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.50.0\">1.50.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/cmp.rs.html#1027-1029\">Source</a></span><a href=\"#method.clamp\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.Ord.html#method.clamp\" class=\"fn\">clamp</a>(self, min: Self, max: Self) -&gt; Self<div class=\"where\">where\n    Self: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/marker/trait.Sized.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::marker::Sized\">Sized</a>,</div></h4></section></summary><div class='docblock'>Restrict a value to a certain interval. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.Ord.html#method.clamp\">Read more</a></div></details></div></details>","Ord","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableGpr","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableXmm"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-PartialEq-for-Writable%3CT%3E\" class=\"impl\"><a href=\"#impl-PartialEq-for-Writable%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialEq.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::cmp::PartialEq\">PartialEq</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialEq.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::cmp::PartialEq\">PartialEq</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.eq\" class=\"method trait-impl\"><a href=\"#method.eq\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialEq.html#tymethod.eq\" class=\"fn\">eq</a>(&amp;self, other: &amp;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Tests for <code>self</code> and <code>other</code> values to be equal, and is used by <code>==</code>.</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.ne\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/cmp.rs.html#261\">Source</a></span><a href=\"#method.ne\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialEq.html#method.ne\" class=\"fn\">ne</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Tests for <code>!=</code>. The default implementation is almost always sufficient,\nand should not be overridden without very good reason.</div></details></div></details>","PartialEq","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableGpr","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableXmm"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-PartialOrd-for-Writable%3CT%3E\" class=\"impl\"><a href=\"#impl-PartialOrd-for-Writable%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialOrd.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::cmp::PartialOrd\">PartialOrd</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialOrd.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::cmp::PartialOrd\">PartialOrd</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.partial_cmp\" class=\"method trait-impl\"><a href=\"#method.partial_cmp\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialOrd.html#tymethod.partial_cmp\" class=\"fn\">partial_cmp</a>(&amp;self, other: &amp;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;) -&gt; <a class=\"enum\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/option/enum.Option.html\" title=\"enum revmc::cranelift::cranelift::prelude::codegen::entity::__core::option::Option\">Option</a>&lt;<a class=\"enum\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/enum.Ordering.html\" title=\"enum revmc::cranelift::cranelift::prelude::codegen::entity::__core::cmp::Ordering\">Ordering</a>&gt;</h4></section></summary><div class='docblock'>This method returns an ordering between <code>self</code> and <code>other</code> values if one exists. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialOrd.html#tymethod.partial_cmp\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.lt\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/cmp.rs.html#1335\">Source</a></span><a href=\"#method.lt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialOrd.html#method.lt\" class=\"fn\">lt</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Tests less than (for <code>self</code> and <code>other</code>) and is used by the <code>&lt;</code> operator. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialOrd.html#method.lt\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.le\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/cmp.rs.html#1353\">Source</a></span><a href=\"#method.le\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialOrd.html#method.le\" class=\"fn\">le</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Tests less than or equal to (for <code>self</code> and <code>other</code>) and is used by the\n<code>&lt;=</code> operator. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialOrd.html#method.le\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.gt\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/cmp.rs.html#1371\">Source</a></span><a href=\"#method.gt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialOrd.html#method.gt\" class=\"fn\">gt</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Tests greater than (for <code>self</code> and <code>other</code>) and is used by the <code>&gt;</code>\noperator. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialOrd.html#method.gt\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.ge\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/cmp.rs.html#1389\">Source</a></span><a href=\"#method.ge\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialOrd.html#method.ge\" class=\"fn\">ge</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Tests greater than or equal to (for <code>self</code> and <code>other</code>) and is used by\nthe <code>&gt;=</code> operator. <a href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.PartialOrd.html#method.ge\">Read more</a></div></details></div></details>","PartialOrd","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableGpr","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableXmm"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-ToWritableReg-for-Writable%3CGpr%3E\" class=\"impl\"><a href=\"#impl-ToWritableReg-for-Writable%3CGpr%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/trait.ToWritableReg.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::ToWritableReg\">ToWritableReg</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/struct.Gpr.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::Gpr\">Gpr</a>&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.to_writable_reg\" class=\"method trait-impl\"><a href=\"#method.to_writable_reg\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/trait.ToWritableReg.html#tymethod.to_writable_reg\" class=\"fn\">to_writable_reg</a>(&amp;self) -&gt; <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Reg.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Reg\">Reg</a>&gt;</h4></section></summary><div class='docblock'>Convert <code>Writable{Xmm,Gpr}</code> to <code>Writable&lt;Reg&gt;</code>.</div></details></div></details>","ToWritableReg","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableGpr"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-ToWritableReg-for-Writable%3CXmm%3E\" class=\"impl\"><a href=\"#impl-ToWritableReg-for-Writable%3CXmm%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/trait.ToWritableReg.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::ToWritableReg\">ToWritableReg</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/struct.Xmm.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::Xmm\">Xmm</a>&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.to_writable_reg\" class=\"method trait-impl\"><a href=\"#method.to_writable_reg\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/isa/x64/args/trait.ToWritableReg.html#tymethod.to_writable_reg\" class=\"fn\">to_writable_reg</a>(&amp;self) -&gt; <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;<a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Reg.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Reg\">Reg</a>&gt;</h4></section></summary><div class='docblock'>Convert <code>Writable{Xmm,Gpr}</code> to <code>Writable&lt;Reg&gt;</code>.</div></details></div></details>","ToWritableReg","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableXmm"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Writable%3CT%3E\" class=\"impl\"><a href=\"#impl-Writable%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.from_reg\" class=\"method\"><h4 class=\"code-header\">pub fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html#tymethod.from_reg\" class=\"fn\">from_reg</a>(reg: T) -&gt; <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;</h4></section></summary><div class=\"docblock\"><p>Explicitly construct a <code>Writable&lt;T&gt;</code> from a <code>T</code>. As noted in\nthe documentation for <code>Writable</code>, this is not hidden or\ndisallowed from the outside; anyone can perform the “cast”;\nbut it is explicit so that we can audit the use sites.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.to_reg\" class=\"method\"><h4 class=\"code-header\">pub fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html#tymethod.to_reg\" class=\"fn\">to_reg</a>(self) -&gt; T</h4></section></summary><div class=\"docblock\"><p>Get the underlying register, which can be read.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.reg_mut\" class=\"method\"><h4 class=\"code-header\">pub fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html#tymethod.reg_mut\" class=\"fn\">reg_mut</a>(&amp;mut self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;mut T</a></h4></section></summary><div class=\"docblock\"><p>Get a mutable borrow of the underlying register.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.map\" class=\"method\"><h4 class=\"code-header\">pub fn <a href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html#tymethod.map\" class=\"fn\">map</a>&lt;U&gt;(self, f: impl <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/ops/trait.Fn.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::ops::Fn\">Fn</a>(T) -&gt; U) -&gt; <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;U&gt;</h4></section></summary><div class=\"docblock\"><p>Map the underlying register to another value or type.</p>\n</div></details></div></details>",0,"revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableGpr","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableXmm"],["<section id=\"impl-Copy-for-Writable%3CT%3E\" class=\"impl\"><a href=\"#impl-Copy-for-Writable%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/marker/trait.Copy.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::marker::Copy\">Copy</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/marker/trait.Copy.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::marker::Copy\">Copy</a>,</div></h3></section>","Copy","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableGpr","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableXmm"],["<section id=\"impl-Eq-for-Writable%3CT%3E\" class=\"impl\"><a href=\"#impl-Eq-for-Writable%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.Eq.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::cmp::Eq\">Eq</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/cmp/trait.Eq.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::cmp::Eq\">Eq</a>,</div></h3></section>","Eq","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableGpr","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableXmm"],["<section id=\"impl-StructuralPartialEq-for-Writable%3CT%3E\" class=\"impl\"><a href=\"#impl-StructuralPartialEq-for-Writable%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T&gt; <a class=\"trait\" href=\"revmc/cranelift/cranelift/prelude/codegen/entity/__core/marker/trait.StructuralPartialEq.html\" title=\"trait revmc::cranelift::cranelift::prelude::codegen::entity::__core::marker::StructuralPartialEq\">StructuralPartialEq</a> for <a class=\"struct\" href=\"revmc/cranelift/cranelift/prelude/codegen/struct.Writable.html\" title=\"struct revmc::cranelift::cranelift::prelude::codegen::Writable\">Writable</a>&lt;T&gt;</h3></section>","StructuralPartialEq","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableGpr","revmc::cranelift::cranelift::prelude::codegen::isa::x64::args::WritableXmm"]]]]);
    if (window.register_type_impls) {
        window.register_type_impls(type_impls);
    } else {
        window.pending_type_impls = type_impls;
    }
})()
//{"start":55,"fragment_lengths":[38351]}