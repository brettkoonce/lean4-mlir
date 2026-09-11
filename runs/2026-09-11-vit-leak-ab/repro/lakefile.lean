import Lake
open Lake DSL
package leakrepro
target readIntoO pkg : System.FilePath := do
  let oFile := pkg.buildDir / "ffi" / "readinto.o"
  let srcJob ← inputTextFile <| pkg.dir / "readinto.c"
  buildO oFile srcJob #["-I", (← getLeanIncludeDir).toString] #["-fPIC", "-O2"]
extern_lib libreadinto pkg := do
  let o ← fetch <| pkg.target ``readIntoO
  buildStaticLib (pkg.staticLibDir / nameToStaticLib "readinto") #[o]
@[default_target]
lean_exe leakrepro where root := `Main
